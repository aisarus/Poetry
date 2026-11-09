#!/usr/bin/env python3
"""Collect public-domain Pushkin poetry passages from Russian Wikisource.

This is a bootstrap corpus of final texts, NOT records of an author's edits.
Each item retains source attribution. Never commit the downloaded JSONL.
"""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

API = 'https://ru.wikisource.org/w/api.php'
CATEGORY = 'Категория:Поэзия Александра Сергеевича Пушкина'
USER_AGENT = 'PoetryLabResearch/0.1 (public-domain text experiment; contact via github.com/aisarus/Poetry)'


class PoemHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get('class', '').split()
        if self.depth:
            if tag in ('br', 'hr', 'img', 'meta', 'link', 'input', 'wbr'):
                if tag == 'br' and not self.skip_depth:
                    self.parts.append('\n')
                return
            self.depth += 1
            if self.skip_depth:
                self.skip_depth += 1
            elif tag in ('style', 'sup', 'script') or 'linenum' in classes:
                self.skip_depth = 1
            elif tag == 'p':
                self.parts.append('\n\n')
        elif tag == 'div' and 'poem' in classes:
            self.depth = 1

    def handle_endtag(self, tag):
        if self.depth:
            if self.skip_depth:
                self.skip_depth -= 1
            if tag == 'p' and not self.skip_depth:
                self.parts.append('\n\n')
            self.depth -= 1

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_data(self, data):
        if self.depth and not self.skip_depth:
            self.parts.append(data)


def parse_poem(html: str) -> str:
    parser = PoemHTML()
    parser.feed(html)
    raw = ''.join(parser.parts).replace('\xa0', ' ')
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in raw.splitlines()]
    lines = [re.sub(r'^\d+\s+', '', line) for line in lines]
    cleaned = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()
    return cleaned


def api(params: dict) -> dict:
    url = API + '?' + urlencode({**params, 'format': 'json', 'formatversion': 2})
    req = Request(url, headers={'User-Agent': USER_AGENT})
    with urlopen(req, timeout=30) as response:
        payload = json.load(response)
    if 'error' in payload:
        raise RuntimeError(f'Викитека API: {payload["error"]}')
    return payload


def titles(max_pages: int) -> list[str]:
    result, continuation = [], {}
    while len(result) < max_pages:
        payload = api({'action': 'query', 'list': 'categorymembers', 'cmtitle': CATEGORY,
                       'cmtype': 'page', 'cmlimit': '50', **continuation})
        for item in payload['query']['categorymembers']:
            title = item['title']
            if title.endswith('(Пушкин)') and '/' not in title:
                result.append(title)
                if len(result) == max_pages:
                    break
        continuation = payload.get('continue', {})
        if not continuation:
            break
    return result


def passages(poem: str, min_lines: int = 8, max_lines: int = 20) -> list[str]:
    lines = [line for line in poem.splitlines() if line.strip()]
    chunks = []
    for start in range(0, len(lines), max_lines):
        part = lines[start:start + max_lines]
        if len(part) >= min_lines:
            chunks.append('\n'.join(part))
    return chunks


def collect(max_pages: int, output: Path, delay: float = 0.2) -> dict:
    if output.exists():
        raise FileExistsError(f'{output}: не перезаписываю готовый корпус')
    output.parent.mkdir(parents=True, exist_ok=True)
    seen, skipped, records = 0, 0, []
    for title in titles(max_pages):
        time.sleep(delay)
        parsed = api({'action': 'parse', 'page': title, 'prop': 'text|categories'})['parse']
        seen += 1
        categories = {item['category'] for item in parsed.get('categories', [])}
        if 'PD-old-70' not in categories:
            skipped += 1
            continue
        source = 'https://ru.wikisource.org/wiki/' + quote(title.replace(' ', '_'))
        for index, text in enumerate(passages(parse_poem(parsed['text'])), 1):
            records.append({'topic': title, 'text': text, 'source_url': source,
                            'passage': index, 'rights_tag': 'PD-old-70'})
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records),
                      encoding='utf-8')
    return {'pages_seen': seen, 'pages_without_PD_tag': skipped,
            'passages': len(records), 'output': str(output)}


def main(argv=None):
    ap = argparse.ArgumentParser(description='Скачать отрывки Пушкина с PD-old-70 из Викитеки')
    ap.add_argument('--max-pages', type=int, default=100)
    ap.add_argument('--out', required=True)
    ap.add_argument('--delay', type=float, default=0.2)
    args = ap.parse_args(argv)
    if args.max_pages < 1 or args.max_pages > 900 or args.delay < 0:
        ap.error('max-pages: 1..900, delay >= 0')
    print(json.dumps(collect(args.max_pages, Path(args.out), args.delay), ensure_ascii=False))


if __name__ == '__main__':
    main()
