"""Conservative Russian stress and rhyme hints.

Unknown stress remains unknown. The transcription is a deliberately small
orthographic approximation, not a full Russian phonological model.
"""
from __future__ import annotations

import re
from pathlib import Path

VOWELS = 'аеёиоуыэюя'
ACCENT = '\u0301'
WORD = re.compile(r'[А-Яа-яЁё]+(?:\u0301?[А-Яа-яЁё]+)*\u0301?')
UNSTRESSED = {'и', 'а', 'но', 'да', 'в', 'во', 'на', 'с', 'со', 'к', 'ко', 'у', 'о',
              'от', 'по', 'за', 'из', 'для', 'не', 'ни', 'же', 'бы', 'ли'}

# Small, auditable seed vocabulary; user-supplied lexicons should cover the
# actual poems. No guessing from word endings when a stress is ambiguous.
SEED = '''
на кры́ше до́ма репети́ровали ма́й
и о́кна разлива́лись акваре́лью
трамва́и па́хли све́жей караме́лью
не прозева́й
вода́ пого́да гора́ го́ре мо́ре вско́ре бу́ря
любо́вь кро́вь зима́ весна́ окно́ давно́
соцве́тий созве́здий возвраще́ний возме́здий
ве́тер све́т со́лнце де́нь но́чь ве́чер у́тро
городско́й го́род сча́стье ра́дость го́лос пе́сня
'''


def _stress_form(form: str) -> tuple[str, int]:
    bare = form.casefold().replace(ACCENT, '')
    vowels = [i for i, c in enumerate(form) if c.casefold() in VOWELS]
    marked = [n for n, i in enumerate(vowels) if form[i] == 'ё' or form[i:i+2].endswith(ACCENT)]
    if len(marked) != 1:
        raise ValueError(f'Нужно ровно одно ударение (´ или ё): {form}')
    return bare, marked[0]


def load_lexicon(path: str | Path | None = None) -> dict[str, int]:
    lex: dict[str, int] = {}
    for token in SEED.split():
        try:
            word, index = _stress_form(token)
            lex[word] = index
        except ValueError:
            pass
    if path is not None:
        for line_no, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            for token in line.split():
                try:
                    word, index = _stress_form(token)
                except ValueError as exc:
                    raise ValueError(f'{path}:{line_no}: {exc}') from exc
                if word in lex and lex[word] != index:
                    # User lexicon intentionally overrides the tiny built-in set.
                    pass
                lex[word] = index
    return lex


def word_stress(token: str, lexicon: dict[str, int]) -> tuple[str, int | None, str]:
    bare = token.casefold().replace(ACCENT, '')
    if ACCENT in token:
        try:
            _, index = _stress_form(token)
            return bare, index, 'explicit'
        except ValueError:
            return bare, None, 'invalid_accent'
    if 'ё' in bare:
        return bare, [i for i, c in enumerate(bare) if c in VOWELS].index(bare.index('ё')), 'yo'
    if bare in lexicon:
        return bare, lexicon[bare], 'lexicon'
    if sum(c in VOWELS for c in bare) == 1:
        return bare, 0, 'monosyllabic'
    return bare, None, 'unknown'


def rhyme_tail(word: str, stressed_vowel: int) -> str:
    """Approximate sound tail, with final devoicing and letter substitutions."""
    positions = [i for i, c in enumerate(word) if c in VOWELS]
    tail = word[positions[stressed_vowel]:]
    for a, b in [('ё', 'о'), ('я', 'а'), ('ю', 'у'), ('е', 'э'), ('й', 'j'),
                 ('ь', ''), ('ъ', '')]:
        tail = tail.replace(a, b)
    # Russian final obstruents are devoiced. This is only a rhyme hint.
    if tail and tail[-1] in 'бвгджз':
        tail = tail[:-1] + dict(zip('бвгджз', 'пфктшс'))[tail[-1]]
    return tail


def analyze_line(line: str, lexicon: dict[str, int]) -> dict:
    entries = []
    pattern = []
    unknown = []
    for match in WORD.finditer(line):
        word, stress, source = word_stress(match.group(), lexicon)
        count = sum(c in VOWELS for c in word)
        if not count:
            continue
        if source == 'unknown' or source == 'invalid_accent':
            unknown.append(word)
        for n in range(count):
            pattern.append('?' if stress is None else ('0' if word in UNSTRESSED else '1') if n == stress else '0')
        entries.append({'word': word, 'syllables': count, 'stress': stress,
                        'source': source})
    return {'words': entries, 'syllables': len(pattern), 'stress_pattern': ''.join(pattern),
            'unknown_stress': unknown}


def compare_rhyme(a: dict, b: dict) -> dict:
    wa, wb = (x['words'][-1] if x['words'] else None for x in (a, b))
    if not wa or not wb:
        return {'status': 'unavailable', 'reason': 'нет конечного слова'}
    if wa['stress'] is None or wb['stress'] is None:
        return {'status': 'unavailable', 'reason': 'неизвестное ударение',
                'words': [wa['word'], wb['word']]}
    tails = [rhyme_tail(w['word'], w['stress']) for w in (wa, wb)]
    return {'status': 'candidate' if tails[0] == tails[1] else 'different',
            'words': [wa['word'], wb['word']], 'sound_tails_approx': tails,
            'note': 'Сравнение орфофонетических окончаний; ручная проверка обязательна.'}
