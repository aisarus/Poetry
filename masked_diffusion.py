#!/usr/bin/env python3
"""Russian poetry masked diffusion: train on revisions, iteratively unmask.

Requires PyTorch and Transformers for train/sample; the pure preparation
functions remain importable without those large optional dependencies.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

LINE, STANZA = '<LINE>', '<STANZA>'
BASE_MODEL = 'ai-forever/ruBert-base'


def read_pairs(path: Path, require_accepted: bool = True) -> list[dict[str, str]]:
    records = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path}:{line_no}: некорректный JSON') from exc
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k].strip()
                                             for k in ('draft', 'revision')):
            raise ValueError(f'{path}:{line_no}: нужны непустые строки draft и revision')
        if 'topic' in item and not isinstance(item['topic'], str):
            raise ValueError(f'{path}:{line_no}: topic должен быть строкой')
        if require_accepted and item.get('accepted') is not True:
            continue
        if item.get('accepted') is False:
            continue
        records.append({'topic': item.get('topic', ''), 'draft': item['draft'],
                        'revision': item['revision']})
    if not records:
        raise ValueError(f'{path}: нет принятых обучающих пар (accepted: true); проверь данные')
    return records


def read_corpus(path: Path) -> list[dict[str, str]]:
    records = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or not isinstance(item.get('text'), str) or not item['text'].strip():
            raise ValueError(f'{path}:{line_no}: нужен непустой text')
        if item.get('rights_tag') != 'PD-old-70' or not str(item.get('source_url', '')).startswith('https://ru.wikisource.org/wiki/'):
            raise ValueError(f'{path}:{line_no}: нет проверенной метки PD и источника')
        records.append({'topic': str(item.get('topic', '')), 'draft': '', 'revision': item['text']})
    if not records:
        raise ValueError(f'{path}: пустой корпус')
    return records


def export_trace(trace_path: Path, topic: str, out: Path) -> int:
    history = json.loads(trace_path.read_text(encoding='utf-8'))
    if not isinstance(history, list):
        raise ValueError('trace.json должен содержать список этапов')
    pairs = []
    for before, after in zip(history, history[1:]):
        if not isinstance(before, dict) or not isinstance(after, dict) or not isinstance(before.get('poem'), str) or not isinstance(after.get('poem'), str):
            raise ValueError('В trace.json отсутствует текст одного из этапов')
        pairs.append({'topic': topic, 'draft': before['poem'], 'revision': after['poem'],
                      'critique': before.get('critique', ''), 'accepted': None})
    if not pairs:
        raise ValueError('Для пар нужна хотя бы одна правка в trace.json')
    if out.exists():
        raise FileExistsError(f'{out}: файл уже существует; не перезаписываю разметку')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in pairs), encoding='utf-8')
    return len(pairs)


def split_poem(poem: str) -> list[str | None]:
    """Lines, with None between stanzas; never publish data via this helper."""
    result: list[str | None] = []
    for block in re.split(r'\n\s*\n', poem.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines:
            if result:
                result.append(None)
            result.extend(lines)
    return result


def target_ids(poem: str, tokenizer) -> tuple[list[int], list[int]]:
    """Return sequence and indices eligible for corruption (not delimiters)."""
    ids, eligible = [], []
    for part in split_poem(poem):
        if part is None:
            ids.append(tokenizer.convert_tokens_to_ids(STANZA))
        else:
            if ids and ids[-1] != tokenizer.convert_tokens_to_ids(STANZA):
                ids.append(tokenizer.convert_tokens_to_ids(LINE))
            words = tokenizer.encode(part, add_special_tokens=False)
            if not words:
                continue
            eligible.extend(range(len(ids), len(ids) + len(words)))
            ids.extend(words)
    return ids, eligible


def prefix_ids(topic: str, draft: str, tokenizer) -> list[int]:
    header = f'Тема: {topic.strip()}\nЧерновик: {draft.strip()}\nРедакция:'
    return [tokenizer.cls_token_id] + tokenizer.encode(header, add_special_tokens=False) + [tokenizer.sep_token_id]


def corrupt(prefix: list[int], target: list[int], eligible: list[int], mask_id: int,
            rng: random.Random, ratio: float) -> tuple[list[int], list[int]]:
    if not eligible or not 0 < ratio <= 1:
        raise ValueError('Нужны маскируемые токены и 0 < ratio <= 1')
    count = max(1, math.ceil(len(eligible) * ratio))
    selected = rng.sample(eligible, count)
    ids = prefix + target[:]
    labels = [-100] * len(ids)
    for pos in selected:
        labels[len(prefix) + pos] = target[pos]
        ids[len(prefix) + pos] = mask_id
    return ids, labels


def scaffold(tokenizer, stanzas: int, tokens_per_line: int, punch_tokens: int,
             draft: str = '', locked_lines: set[int] | None = None) -> tuple[list[int], set[int]]:
    if not 1 <= stanzas <= 12 or not 1 <= punch_tokens <= tokens_per_line <= 64:
        raise ValueError('stanzas: 1..12; 1 <= punch_tokens <= tokens_per_line <= 64')
    locked_lines = locked_lines or set()
    draft_lines = [part for part in split_poem(draft) if part is not None]
    if locked_lines and (not draft or len(draft_lines) != 4 * stanzas
                         or min(locked_lines) < 1 or max(locked_lines) > 4 * stanzas):
        raise ValueError('Для --lock-lines нужен черновик с 4 строками на строфу')
    ids, editable = [], set()
    for stanza in range(stanzas):
        if stanza:
            ids.append(tokenizer.convert_tokens_to_ids(STANZA))
        for line in range(4):
            if line:
                ids.append(tokenizer.convert_tokens_to_ids(LINE))
            number = stanza * 4 + line + 1
            if number in locked_lines:
                ids.extend(tokenizer.encode(draft_lines[number - 1], add_special_tokens=False))
            else:
                count = punch_tokens if line == 3 else tokens_per_line
                editable.update(range(len(ids), len(ids) + count))
                ids.extend([tokenizer.mask_token_id] * count)
    return ids, editable


def render(ids: list[int], tokenizer) -> str:
    lines = []
    current = []
    def flush():
        if current:
            lines.append(tokenizer.decode(current, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=True).strip())
            current.clear()
    for token in ids:
        if token == tokenizer.convert_tokens_to_ids(STANZA):
            flush()
            lines.append('')
        elif token == tokenizer.convert_tokens_to_ids(LINE):
            flush()
        else:
            current.append(token)
    flush()
    return '\n'.join(lines).strip()


def choose_commit(confidences: dict[int, float], remaining_steps: int) -> list[int]:
    """Confidence-ranked reverse transition; each masked site commits once."""
    if remaining_steps < 1:
        raise ValueError('remaining_steps должен быть положительным')
    amount = math.ceil(len(confidences) / remaining_steps)
    return sorted(confidences, key=lambda p: (-confidences[p], p))[:amount]


def reverse_unmask(sequence: list[int], prefix_len: int, editable: set[int],
                   mask_id: int, steps: int, predictor) -> list[int]:
    """Absorbing reverse process; predictor returns {position: (token, confidence)}."""
    if steps < 1:
        raise ValueError('steps должен быть положительным')
    for step in range(steps):
        masked = [p for p in sorted(editable) if sequence[prefix_len + p] == mask_id]
        if not masked:
            break
        choices = predictor(sequence, masked)
        if set(choices) != set(masked) or any(token == mask_id for token, _ in choices.values()):
            raise ValueError('Предсказатель должен заполнить все маски допустимыми токенами')
        for p in choose_commit({p: choices[p][1] for p in masked}, steps - step):
            sequence[prefix_len + p] = choices[p][0]
    if any(sequence[prefix_len + p] == mask_id for p in editable):
        raise RuntimeError('Обратный процесс оставил маски')
    return sequence


def _dependencies():
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Установи зависимости: pip install -r requirements-diffusion.txt') from exc
    return torch, AutoTokenizer, AutoModelForMaskedLM


def train(args) -> None:
    torch, AutoTokenizer, AutoModelForMaskedLM = _dependencies()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    records = (read_pairs(Path(args.data), require_accepted=not args.include_unreviewed)
               if args.data else read_corpus(Path(args.corpus)))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.mask_token_id is None or tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError('Нужен BERT-подобный токенизатор с MASK/CLS/SEP')
    tokenizer.add_special_tokens({'additional_special_tokens': [LINE, STANZA]})
    model = AutoModelForMaskedLM.from_pretrained(args.base_model)
    model.resize_token_embeddings(len(tokenizer))
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError(f'max_length превышает предел модели: {model.config.max_position_embeddings}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if device == 'cuda':
        model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device == 'cuda')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    seen = skipped = updates = 0
    for epoch in range(args.epochs):
        rng.shuffle(records)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for record in records[:args.limit or None]:
            # Learn both revision with visible draft and topic-only generation.
            draft = '' if rng.random() < args.draft_dropout else record['draft']
            prefix = prefix_ids(record['topic'], draft, tokenizer)
            target, eligible = target_ids(record['revision'], tokenizer)
            if not eligible or len(prefix) + len(target) + 1 > args.max_length:
                skipped += 1
                continue
            ratio = rng.uniform(0.05, 1.0)
            ids, labels = corrupt(prefix, target, eligible, tokenizer.mask_token_id, rng, ratio)
            ids.append(tokenizer.sep_token_id)
            labels.append(-100)
            inputs = torch.tensor([ids], device=device)
            targets = torch.tensor([labels], device=device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device == 'cuda'):
                loss = model(input_ids=inputs, labels=targets).loss
            scaler.scale(loss / args.accumulation).backward()
            pending += 1
            seen += 1
            if pending == args.accumulation:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                pending = 0
        if pending:
            # Compensate for the last smaller accumulation group.
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(args.accumulation / pending)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        if not seen:
            raise ValueError('Все примеры превышают max_length или пусты; модель не обучалась')
        print(f'epoch {epoch + 1}/{args.epochs}: seen={seen} skipped={skipped} updates={updates}', flush=True)
        model.save_pretrained(output)
        tokenizer.save_pretrained(output)
        (output / 'diffusion_config.json').write_text(json.dumps({
            'architecture': 'absorbing_mask_discrete_diffusion', 'base_model': args.base_model,
            'max_length': args.max_length, 'epochs_completed': epoch + 1,
            'seen_examples': seen, 'skipped_examples': skipped, 'optimizer_updates': updates,
            'draft_dropout': args.draft_dropout, 'seed': args.seed,
        }, ensure_ascii=False, indent=2), encoding='utf-8')


def sample(args) -> str:
    torch, AutoTokenizer, AutoModelForMaskedLM = _dependencies()
    checkpoint = Path(args.checkpoint)
    if not (checkpoint / 'diffusion_config.json').exists():
        raise ValueError('Нужен обученный чекпойнт Poetry Lab с diffusion_config.json')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    draft = Path(args.draft_file).read_text(encoding='utf-8-sig') if args.draft_file else ''
    locked = {int(x) for x in args.lock_lines.split(',') if x.strip()} if args.lock_lines else set()
    target, editable = scaffold(tokenizer, args.stanzas, args.tokens_per_line,
                                args.punch_tokens, draft, locked)
    prefix = prefix_ids(args.topic, draft, tokenizer)
    max_length = json.loads((checkpoint / 'diffusion_config.json').read_text(encoding='utf-8'))['max_length']
    if len(prefix) + len(target) + 1 > max_length:
        raise ValueError(f'Длина {len(prefix) + len(target) + 1} превышает max_length={max_length}; сократи черновик/число токенов')
    sequence = prefix + target + [tokenizer.sep_token_id]
    blocked = set(tokenizer.all_special_ids)
    blocked.update([tokenizer.unk_token_id, tokenizer.mask_token_id])
    blocked.discard(None)
    def predict(current, masked):
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([current], device=device)).logits[0]
        choices = {}
        for p in masked:
            row = logits[len(prefix) + p].float().clone()
            row[list(blocked)] = -float('inf')
            top = torch.topk(row, k=min(args.top_k, row.numel()))
            probabilities = torch.softmax(top.values / args.temperature, dim=0)
            pick = torch.multinomial(probabilities, 1).item()
            choices[p] = int(top.indices[pick]), float(probabilities[pick])
        return choices
    sequence = reverse_unmask(sequence, len(prefix), editable, tokenizer.mask_token_id,
                              args.steps, predict)
    return render(sequence[len(prefix):len(prefix) + len(target)], tokenizer)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Обучение и генерация masked diffusion Poetry Lab')
    sub = parser.add_subparsers(dest='command', required=True)
    tr = sub.add_parser('train')
    source = tr.add_mutually_exclusive_group(required=True)
    source.add_argument('--data', help='JSONL: topic, draft, revision, accepted')
    source.add_argument('--corpus', help='JSONL с финальными стихами и проверенными PD метками для bootstrap')
    tr.add_argument('--out', required=True, help='Папка для весов и токенизатора (не коммитить)')
    tr.add_argument('--base-model', default=BASE_MODEL)
    tr.add_argument('--epochs', type=int, default=3)
    tr.add_argument('--max-length', type=int, default=256)
    tr.add_argument('--limit', type=int, default=0)
    tr.add_argument('--accumulation', type=int, default=8)
    tr.add_argument('--lr', type=float, default=2e-5)
    tr.add_argument('--draft-dropout', type=float, default=0.25)
    tr.add_argument('--seed', type=int, default=42)
    tr.add_argument('--include-unreviewed', action='store_true', help='Эксперимент: учить на правках без человеческого accepted: true')
    ex = sub.add_parser('export-trace', help='Из Ollama trace.json создать пары для ручного отбора')
    ex.add_argument('--trace', required=True)
    ex.add_argument('--topic', required=True)
    ex.add_argument('--out', required=True)
    sa = sub.add_parser('sample')
    sa.add_argument('--checkpoint', required=True)
    sa.add_argument('--topic', required=True)
    sa.add_argument('--draft-file', help='Частично переработать собственный черновик')
    sa.add_argument('--lock-lines', default='', help='Сохранить строки черновика, например 1,4')
    sa.add_argument('--stanzas', type=int, default=2)
    sa.add_argument('--tokens-per-line', type=int, default=10)
    sa.add_argument('--punch-tokens', type=int, default=4)
    sa.add_argument('--steps', type=int, default=16)
    sa.add_argument('--top-k', type=int, default=8)
    sa.add_argument('--temperature', type=float, default=0.8)
    sa.add_argument('--seed', type=int, default=42)
    sa.add_argument('--out', help='Файл результата; по умолчанию только stdout')
    args = parser.parse_args(argv)
    if args.command == 'train':
        if args.epochs < 1 or args.max_length < 32 or args.accumulation < 1 or args.lr <= 0 or not 0 <= args.draft_dropout <= 1 or args.limit < 0:
            parser.error('Некорректные параметры обучения')
        if args.corpus and args.include_unreviewed:
            parser.error('--include-unreviewed применяется только к --data')
        train(args)
    elif args.command == 'export-trace':
        print(f'Экспортировано пар: {export_trace(Path(args.trace), args.topic, Path(args.out))}. Отметь удачные accepted: true после просмотра.')
    else:
        if args.steps < 1 or args.top_k < 1 or args.temperature <= 0:
            parser.error('steps/top-k/temperature должны быть положительными')
        poem = sample(args)
        if args.out:
            destination = Path(args.out)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(poem + '\n', encoding='utf-8')
        print(poem)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
