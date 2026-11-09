#!/usr/bin/env python3
"""Poetry Lab MVP: local drafting, analysis and iterative revision via Ollama.

No Python third-party dependencies. This is NOT a diffusion model or an
objective poetry-quality evaluator. The prosody diagnostics are heuristics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any
import phonetics
import audio_feedback

VOWELS = 'аеёиоуыэюя'
VOWELS_ALL = VOWELS + VOWELS.upper()
WORD_RE = re.compile(r"[А-Яа-яЁё]+(?:[-’'][А-Яа-яЁё]+)*|[A-Za-z]+(?:[-’'][A-Za-z]+)*")
STANZA_SPLIT = re.compile(r"\n\s*\n+")
END_PUNCT = re.compile(r'[\s\.,!?…:;\-—«»"„“”()]+$')
ACCENT = '\u0301'
DEFAULT_SCHEMES = ['ABBA', 'ABAB', 'AABB']


def syllables(line: str) -> int:
    return sum(c in VOWELS_ALL for c in line)


def words(line: str) -> list[str]:
    return WORD_RE.findall(line)


def ending_word(line: str) -> str:
    hits = words(END_PUNCT.sub('', line))
    return hits[-1].casefold() if hits else ''


def vowel_signature(word: str) -> str:
    """Only a rough, non-stress-aware signature: not a phonemic rhyme test."""
    return ''.join(c for c in word if c in VOWELS)


def rhyme_hint(a: str, b: str) -> dict[str, Any]:
    wa, wb = ending_word(a), ending_word(b)
    va, vb = vowel_signature(wa), vowel_signature(wb)
    shared = 0
    for x, y in zip(reversed(va), reversed(vb)):
        if x != y:
            break
        shared += 1
    # Compare final few written letters as a weak secondary hint.
    tail = 0
    for x, y in zip(reversed(wa), reversed(wb)):
        if x != y:
            break
        tail += 1
    return {'words': [wa, wb], 'shared_final_vowels': shared,
            'shared_final_letters': tail, 'tentative': True}


def clean_poem(text: str) -> str:
    """Remove Markdown fences/headings, if the model wraps its response."""
    t = text.strip()
    if t.startswith('```'):
        t = re.sub(r'^```[^\n]*\n', '', t)
        t = re.sub(r'\n```\s*$', '', t)
    return t.strip()


def analyze(poem: str, schemes: list[str] | None = None,
            lexicon: dict[str, int] | None = None) -> dict[str, Any]:
    schemes = schemes or DEFAULT_SCHEMES
    lexicon = lexicon if lexicon is not None else phonetics.load_lexicon()
    blocks = [b for b in STANZA_SPLIT.split(poem.strip()) if b.strip()]
    result: list[dict[str, Any]] = []
    for i, block in enumerate(blocks):
        lines = [x.strip() for x in block.splitlines() if x.strip()]
        sc = schemes[i % len(schemes)]
        counts = [syllables(x) for x in lines]
        phonetic_lines = [phonetics.analyze_line(x, lexicon) for x in lines]
        core = counts[:3]
        # Structural feedback, not a metric quality score.
        long_span = max(core) - min(core) if len(core) == 3 else None
        short_ok = 1 <= counts[3] <= 5 if len(counts) == 4 else False
        groups: dict[str, list[int]] = {}
        for j, label in enumerate(sc):
            groups.setdefault(label, []).append(j)
        rhymes = []
        for positions in groups.values():
            if len(positions) == 2 and max(positions) < len(lines):
                x, y = positions
                rhymes.append({'line_indices': [x + 1, y + 1],
                               **rhyme_hint(lines[x], lines[y]),
                               'phonetic': phonetics.compare_rhyme(phonetic_lines[x], phonetic_lines[y])})
        result.append({
            'number': i + 1, 'expected_scheme': sc,
            'lines': lines, 'line_count': len(lines), 'syllables': counts,
            'phonetic_lines': phonetic_lines,
            'three_long_lines_span': long_span,
            'short_fourth_line_1_to_5_syllables': short_ok,
            'rhyme_hints': rhymes,
            'warnings': ([f'ожидалось 4 строки, найдено {len(lines)}'] if len(lines) != 4 else [])
                        + (['последняя строка длиннее 5 слогов'] if len(lines) == 4 and not short_ok else [])
                        + (['первые 3 строки сильно различаются по числу слогов']
                           if long_span is not None and long_span >= 4 else []),
        })
    return {
        'stanzas': result, 'stanza_count': len(result),
        'disclaimer': 'Ударения известны лишь для словаря, ё и явных акцентов. Орфофонетические окончания — подсказка, не полная G2P или проверка метра.',
    }


def ollama_chat(messages: list[dict[str, str]], model: str, base_url: str, temperature: float = 0.7) -> str:
    endpoint = base_url.rstrip('/') + '/api/chat'
    data = json.dumps({'model': model, 'messages': messages, 'stream': False,
                       'options': {'temperature': temperature}}).encode('utf-8')
    req = urllib.request.Request(endpoint, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode('utf-8', errors='replace')[:500]
        raise RuntimeError(f'Ollama HTTP {exc.code}: {details}') from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f'Не удалось обратиться к Ollama ({endpoint}): {exc}') from exc
    content = payload.get('message', {}).get('content', '')
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError('Ollama вернула пустое сообщение.')
    return content.strip()


def draft_prompt(topic: str, stanzas: int, schemes: list[str]) -> str:
    seq = [schemes[i % len(schemes)] for i in range(stanzas)]
    return f'''Напиши русское стихотворение на тему: {topic}.
Требования: {stanzas} четверостиший, схема по строфам: {', '.join(seq)}.
В КАЖДОЙ строфе первые три строки стремятся к одинаковому ритму и длине, четвертая — короткая (1–5 слогов) и смысловой панч.
Рифмуй по звучанию и реальному ударению, не по орфографии. Используй ассонансы в 2–3 гласных звука, внутренние рифмы (минимум 2) и осмысленные необычные метафоры. Позитивный тон без открыток и общих слов.
Приоритеты: художественная цельность, узнаваемое развитие образов, читаемость вслух. Не пиши бессмысленных сравнений, случайных красивых слов и набора картинок.
Отдай ТОЛЬКО стихотворение: без заголовка, списка и пояснений.'''


def critic_prompt(poem: str, report: dict[str, Any], audio: dict | None = None) -> str:
    compact = [{'number': s['number'], 'syllables': s['syllables'],
                'long_span': s['three_long_lines_span'], 'expected_scheme': s['expected_scheme'],
                'stress_patterns': [p['stress_pattern'] for p in s['phonetic_lines']],
                'unknown_stress': [p['unknown_stress'] for p in s['phonetic_lines']],
                'rhyme_hints': [r['phonetic'] for r in s['rhyme_hints']]}
               for s in report['stanzas']]
    return f'''Ты строгий литературный редактор и профессиональный чтец русской поэзии.
Разбери СЛЕДУЮЩИЙ текст: (а) где при декламации сбиваются ударения/ритм, (б) неуместные метафоры и логические разрывы, (в) слабые окончания, (г) композиция и развитие мысли, (д) какие строки нельзя менять, потому что они удачны.
Это не аудиопрослушивание: не утверждай, будто ты реально услышал звук. Анализируй предполагаемую декламацию по тексту. Не считай число слогов эквивалентом метра. Будь конкретен, приводи проблемные строки.
Автоматическая неполная фонетическая сводка: {json.dumps(compact, ensure_ascii=False)}
Аудионаблюдения (только если есть WAV): {json.dumps(audio, ensure_ascii=False) if audio is not None else 'нет; не утверждай, что слышал чтение'}
Если есть аудио, опирайся только на измеренные длительности, паузы и примерную высоту; не придумывай услышанные интонации, ударения или дикцию. Сравни темп первых трёх строк каждой строфы, учитывая число слогов и ограничения синтетического голоса.
ТЕКСТ:\n{poem}
Ответь кратко: максимум 12 конкретных замечаний, без переписывания стихотворения.'''


def revision_prompt(poem: str, critique: str, report: dict[str, Any]) -> str:
    short = [{'number': x['number'], 'syllables': x['syllables'],
              'short_line_ok': x['short_fourth_line_1_to_5_syllables'],
              'scheme': x['expected_scheme']} for x in report['stanzas']]
    return f'''Сделай редактуру стихотворения, используя критику ниже. Сохрани замысел, основные образы, удачные строки и {len(report['stanzas'])} строф, каждая 4 строки.
Схемы рифмовки, по строфам: {', '.join(x['scheme'] for x in short)}.
Три длинных строки в каждой строфе должны быть ритмически согласованы, последняя — до 5 слогов, смысловой панч.
НЕ заменяй конкретные образы на абстрактные «свет, тьма, вечность» и не добавляй бессмысленных синестезий. Если критика неверна, не следуй ей. Рифмуй на слух, проверяй ударения.
ТЕХНИЧЕСКИЕ ПОДСКАЗКИ (приблизительные): {json.dumps(short, ensure_ascii=False)}
КРИТИКА:\n{critique}
ИСХОДНЫЙ ТЕКСТ:\n{poem}
Верни только полностью исправленное стихотворение.'''


def speak_windows(poem: str, output_wav: Path) -> None:
    """Windows SAPI: offline after OS voice installation; voice/prosody vary."""
    if os.name != 'nt':
        raise RuntimeError('В этой версии запись WAV реализована только для Windows (SAPI).')
    output_wav = output_wav.resolve()
    # Avoid interpolation of user text into the PowerShell script: pass file path instead.
    text_path = output_wav.with_suffix('.speak.txt')
    text_path.write_text(poem, encoding='utf-8-sig')
    script = '''param([string]$inputPath, [string]$outputPath)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $ru = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'ru-*' } | Select-Object -First 1
  if ($ru) { $s.SelectVoice($ru.VoiceInfo.Name) } else { Write-Warning 'Русский голос SAPI не найден. Речь будет прочитана голосом по умолчанию.' }
  $s.SetOutputToWaveFile($outputPath)
  $raw = [System.IO.File]::ReadAllText($inputPath, [System.Text.Encoding]::UTF8)
  $s.Speak($raw)
} finally { $s.Dispose() }
'''
    ps_path = output_wav.with_suffix('.speak.ps1')
    ps_path.write_text(script, encoding='utf-8-sig')
    try:
        subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                        '-File', str(ps_path), str(text_path), str(output_wav)], check=True)
    except FileNotFoundError as exc:
        raise RuntimeError('Не найден powershell.exe.') from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('Синтез речи не удался: проверь наличие голосов Windows SAPI.') from exc
    finally:
        text_path.unlink(missing_ok=True)
        ps_path.unlink(missing_ok=True)


def speak_windows_batch(poem: str, run_dir: Path, step: int) -> tuple[Path, list[Path]]:
    """SAPI reads the full poem and each line in one process for acoustic analysis."""
    if os.name != 'nt':
        raise RuntimeError('--audio-feedback требует Windows SAPI или тестовый синтезатор.')
    audio_dir = run_dir / f'audio_{step:02d}'
    audio_dir.mkdir(parents=True, exist_ok=True)
    full = run_dir / f'reading_{step:02d}.wav'
    lines = [line.strip() for line in poem.splitlines() if line.strip()]
    paths = [audio_dir / f'line_{i:03d}.wav' for i in range(1, len(lines) + 1)]
    payload = audio_dir / 'input.json'
    ps_path = audio_dir / 'speak.ps1'
    payload.write_text(json.dumps({'poem': poem, 'lines': lines}, ensure_ascii=False), encoding='utf-8')
    ps_path.write_text('''param([string]$inputPath, [string]$fullPath, [string]$audioDir)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$data = Get-Content -LiteralPath $inputPath -Raw -Encoding UTF8 | ConvertFrom-Json
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $ru = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'ru-*' } | Select-Object -First 1
  if (-not $ru) { throw 'Русский голос Windows SAPI не найден' }
  $s.SelectVoice($ru.VoiceInfo.Name)
  $s.SetOutputToWaveFile($fullPath)
  $s.Speak([string]$data.poem)
  $s.SetOutputToNull()
  for ($i = 0; $i -lt $data.lines.Count; $i++) {
    $path = Join-Path $audioDir ('line_{0:D3}.wav' -f ($i + 1))
    $s.SetOutputToWaveFile($path)
    $s.Speak([string]$data.lines[$i])
    $s.SetOutputToNull()
  }
} finally { $s.Dispose() }
''', encoding='utf-8-sig')
    try:
        subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                        '-File', str(ps_path), str(payload), str(full), str(audio_dir)], check=True)
        if not full.is_file() or any(not path.is_file() for path in paths):
            raise RuntimeError('SAPI не создал все WAV-файлы.')
        return full, paths
    except FileNotFoundError as exc:
        raise RuntimeError('Не найден powershell.exe.') from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('Синтез SAPI не удался; проверь русский голос Windows.') from exc
    finally:
        payload.unlink(missing_ok=True)
        ps_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Poetry Lab: Ollama + phonetic hints + optional WAV feedback')
    ap.add_argument('--topic', default='Радость неожиданного человеческого взаимопонимания')
    ap.add_argument('--file', help='Проанализировать/редактировать существующий UTF-8 текст вместо генерации')
    ap.add_argument('--model', default='qwen2.5:7b')
    ap.add_argument('--url', default='http://127.0.0.1:11434')
    ap.add_argument('--stanzas', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=2, help='Число итераций критика→редактор. 0 — только анализ черновика.')
    ap.add_argument('--schemes', default='ABBA,ABAB,AABB')
    ap.add_argument('--stress-lexicon', help='UTF-8 словарь ударений, формы вроде караме́лью, одна или несколько на строку')
    ap.add_argument('--out', default='output')
    ap.add_argument('--speak', action='store_true', help='Сделать WAV с помощью Windows SAPI (не оценка звука!)')
    ap.add_argument('--audio-feedback', action='store_true', help='Синтезировать и измерять WAV на каждой итерации перед критикой (Windows, русский SAPI голос)')
    ap.add_argument('--analyze-only', action='store_true', help='Только статическая разметка, без Ollama; требует --file')
    args = ap.parse_args(argv)
    if args.stanzas < 1 or args.stanzas > 30 or args.rounds < 0 or args.rounds > 20:
        ap.error('stanzas: 1..30; rounds: 0..20')
    schemes = [x.strip().upper() for x in args.schemes.split(',') if x.strip()]
    if not schemes or any(len(x) != 4 or set(x) != {'A', 'B'} or x.count('A') != 2 or x.count('B') != 2 for x in schemes):
        ap.error('Каждая рифмовка должна иметь вид AABB, ABAB или ABBA.')
    if args.analyze_only and not args.file:
        ap.error('--analyze-only требует --file')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    try:
        lexicon = phonetics.load_lexicon(args.stress_lexicon)
        if args.file:
            current = Path(args.file).read_text(encoding='utf-8-sig').strip()
        else:
            print('Создание черновика...', flush=True)
            current = clean_poem(ollama_chat([
                {'role': 'system', 'content': 'Ты русскоязычный поэт, внимательно работающий с ритмом, рифмой и точной метафорой.'},
                {'role': 'user', 'content': draft_prompt(args.topic, args.stanzas, schemes)}
            ], args.model, args.url, 0.8))
        for step in range(args.rounds + 1 if not args.analyze_only else 1):
            report = analyze(current, schemes, lexicon)
            (run_dir / f'poem_{step:02d}.txt').write_text(current + '\n', encoding='utf-8')
            audio_report = None
            if args.audio_feedback:
                full_wav, line_wavs = speak_windows_batch(current, run_dir, step)
                phonetic_lines = [p for stanza in report['stanzas'] for p in stanza['phonetic_lines']]
                text_lines = [line for stanza in report['stanzas'] for line in stanza['lines']]
                audio_report = audio_feedback.reading_report(text_lines, line_wavs,
                                                              [p['syllables'] for p in phonetic_lines])
                audio_report['full_wav'] = str(full_wav)
                (run_dir / f'audio_metrics_{step:02d}.json').write_text(
                    json.dumps(audio_report, ensure_ascii=False, indent=2), encoding='utf-8')
                report['audio_feedback'] = audio_report
            (run_dir / f'metrics_{step:02d}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'\nЭтап {step}: строф={report["stanza_count"]}, слоги=',
                  [r['syllables'] for r in report['stanzas']], flush=True)
            item: dict[str, Any] = {'step': step, 'poem': current, 'prosody_report': report}
            if step == args.rounds or args.analyze_only:
                history.append(item)
                break
            print('Критик анализирует смысл и предполагаемое звучание...', flush=True)
            critique = ollama_chat([
                {'role': 'system', 'content': 'Ты беспристрастный строгий редактор. Не льсти и не придумывай несуществовавшего прослушивания аудио.'},
                {'role': 'user', 'content': critic_prompt(current, report, audio_report)},
            ], args.model, args.url, 0.25)
            item['critique'] = critique
            (run_dir / f'critique_{step:02d}.txt').write_text(critique + '\n', encoding='utf-8')
            history.append(item)
            print('Редактор создаёт новую версию...', flush=True)
            current = clean_poem(ollama_chat([
                {'role': 'system', 'content': 'Ты редактор стихов. Береги художественную цельность и проверяй звук, ударения, рифмы.'},
                {'role': 'user', 'content': revision_prompt(current, critique, report)},
            ], args.model, args.url, 0.6))
        (run_dir / 'trace.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
        (run_dir / 'final.txt').write_text(current + '\n', encoding='utf-8')
        if args.speak and not args.audio_feedback:
            speak_windows(current, run_dir / 'final.wav')
        elif args.speak and args.audio_feedback:
            (run_dir / 'final.wav').write_bytes((run_dir / f'reading_{args.rounds if not args.analyze_only else 0:02d}.wav').read_bytes())
        print(f'\nСохранено: {run_dir.resolve()}')
        print('\n' + current)
        return 0
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f'ОШИБКА: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
