#!/usr/bin/env python3
"""Prepare blinded A/B poems and score completed human preferences."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{n}: некорректный JSON') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{n}: ожидается объект')
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                    encoding='utf-8')


def prepare(candidates: Path, out_dir: Path, seed: int = 42) -> int:
    rows = read_jsonl(candidates)
    if not rows:
        raise ValueError('Нет кандидатов для слепой оценки')
    ids = set()
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip()
               for k in ('topic_id', 'topic', 'baseline', 'diffusion')):
            raise ValueError('Нужны непустые строки topic_id, topic, baseline, diffusion')
        if row['topic_id'] in ids or row['baseline'].strip() == row['diffusion'].strip():
            raise ValueError(f'Повтор или одинаковые варианты для {row["topic_id"]}')
        ids.add(row['topic_id'])
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f'{out_dir}: папка не пуста; не перезаписываю оценку')
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rng.shuffle(rows)
    ballot, key = [], {}
    for row in rows:
        baseline_a = bool(rng.getrandbits(1))
        a = row['baseline'] if baseline_a else row['diffusion']
        b = row['diffusion'] if baseline_a else row['baseline']
        ballot.append({'topic_id': row['topic_id'], 'topic': row['topic'], 'A': a,
                       'B': b, 'preferred': None, 'notes': ''})
        key[row['topic_id']] = {'A': 'baseline' if baseline_a else 'diffusion',
                                'B': 'diffusion' if baseline_a else 'baseline'}
    write_jsonl(out_dir / 'ballot.jsonl', ballot)
    (out_dir / 'key.json').write_text(json.dumps(key, ensure_ascii=False, indent=2),
                                      encoding='utf-8')
    return len(rows)


def wilson95(wins: int, n: int) -> list[float] | None:
    if not n:
        return None
    z = 1.959963984540054
    p = wins / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(center - radius, 4), round(center + radius, 4)]


def score(ballot_path: Path, key_path: Path) -> dict:
    ballot = read_jsonl(ballot_path)
    key = json.loads(key_path.read_text(encoding='utf-8'))
    if not ballot or len(ballot) != len(key) or {r.get('topic_id') for r in ballot} != set(key):
        raise ValueError('Бюллетень и ключ содержат разные темы')
    counts = {'diffusion': 0, 'baseline': 0, 'tie': 0}
    for row in ballot:
        choice = row.get('preferred')
        if choice not in ('A', 'B', 'tie'):
            raise ValueError(f'{row["topic_id"]}: сначала отметь preferred: A, B или tie')
        if choice == 'tie':
            counts['tie'] += 1
        else:
            counts[key[row['topic_id']][choice]] += 1
    decisive = counts['diffusion'] + counts['baseline']
    return {'n': len(ballot), 'diffusion_wins': counts['diffusion'],
            'baseline_wins': counts['baseline'], 'ties': counts['tie'],
            'diffusion_share_of_decisive': round(counts['diffusion'] / decisive, 4) if decisive else None,
            'wilson95_decisive': wilson95(counts['diffusion'], decisive),
            'warning': 'Интервал описывает неопределённость предпочтений в этой выборке; качество и независимость тем проверяет человек.'}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Слепая попарная оценка диффузии против baseline')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--candidates', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    s = sub.add_parser('score')
    s.add_argument('--ballot', required=True)
    s.add_argument('--key', required=True)
    s.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'prepare':
        print(f'Подготовлено {prepare(Path(args.candidates), Path(args.out), args.seed)} слепых пар. Спрячь key.json до заполнения ballot.jsonl.')
    else:
        result = score(Path(args.ballot), Path(args.key))
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
