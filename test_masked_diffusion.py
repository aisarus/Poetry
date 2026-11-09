import json
import random
import tempfile
import unittest
from pathlib import Path

import masked_diffusion as md


class TinyTokenizer:
    cls_token_id, sep_token_id, mask_token_id = 1, 2, 3
    def __init__(self):
        self.vocab = {md.LINE: 4, md.STANZA: 5}
        self.reverse = {4: md.LINE, 5: md.STANZA}

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def encode(self, text, add_special_tokens=False):
        result = []
        for word in text.split():
            if word not in self.vocab:
                token = len(self.vocab) + 6
                self.vocab[word] = token
                self.reverse[token] = word
            result.append(self.vocab[word])
        return result

    def decode(self, ids, **kwargs):
        return ' '.join(self.reverse[x] for x in ids)


class DiffusionTests(unittest.TestCase):
    def test_corruption_only_masks_poem_words_and_labels_same_tokens(self):
        tok = TinyTokenizer()
        target, eligible = md.target_ids('На крыше\nМай\n\nНе прозевай', tok)
        prefix = md.prefix_ids('Тема', 'Черновик', tok)
        ids, labels = md.corrupt(prefix, target, eligible, tok.mask_token_id,
                                 random.Random(4), 0.5)
        masked = [i for i, token in enumerate(ids) if token == tok.mask_token_id]
        self.assertEqual(len(masked), 3)
        self.assertTrue(all(labels[i] == (prefix + target)[i] for i in masked))
        self.assertEqual([ids[len(prefix) + i] for i, x in enumerate(target) if i not in eligible],
                         [x for i, x in enumerate(target) if i not in eligible])
        self.assertEqual(md.render(target, tok), 'На крыше\nМай\n\nНе прозевай')

    def test_scaffold_locks_lines_and_reverse_schedule_finishes(self):
        tok = TinyTokenizer()
        draft = 'Первая строка\nВторая строка\nТретья строка\nПанч'
        ids, editable = md.scaffold(tok, 1, 3, 1, draft, {1, 4})
        self.assertTrue(all(ids[p] == tok.mask_token_id for p in editable))
        self.assertIn('Первая строка', md.render(ids[:2], tok))
        remaining = set(editable)
        for step in range(4):
            committed = md.choose_commit({p: 0.5 + p / 100 for p in remaining}, 4 - step)
            remaining.difference_update(committed)
        self.assertFalse(remaining)
        self.assertRaises(ValueError, md.scaffold, tok, 1, 3, 1, '', {1})
        prefix = [tok.cls_token_id, tok.sep_token_id]
        result = md.reverse_unmask(prefix + ids, len(prefix), editable, tok.mask_token_id,
                                   4, lambda current, missing: {p: (100 + p, float(p)) for p in missing})
        self.assertEqual([result[len(prefix) + p] for p in sorted(editable)],
                         [100 + p for p in sorted(editable)])
        self.assertEqual(result[len(prefix):len(prefix) + 2], ids[:2])

    def test_trace_export_requires_human_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'trace.json'
            dest = Path(tmp) / 'pairs.jsonl'
            source.write_text(json.dumps([{'poem': 'Черновик', 'critique': 'замечание'},
                                          {'poem': 'Правка'}], ensure_ascii=False), encoding='utf-8')
            self.assertEqual(md.export_trace(source, 'Тема', dest), 1)
            self.assertRaisesRegex(ValueError, 'нет принятых', md.read_pairs, dest)
            row = json.loads(dest.read_text(encoding='utf-8'))
            self.assertIsNone(row['accepted'])
            row['accepted'] = True
            dest.write_text(json.dumps(row, ensure_ascii=False), encoding='utf-8')
            self.assertEqual(md.read_pairs(dest)[0]['revision'], 'Правка')
            self.assertRaises(FileExistsError, md.export_trace, source, 'Тема', dest)

    def test_bootstrap_corpus_requires_rights_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'corpus.jsonl'
            path.write_text(json.dumps({'text': 'Строки стиха', 'topic': 'Пушкин',
                                        'source_url': 'https://ru.wikisource.org/wiki/Test',
                                        'rights_tag': 'PD-old-70'}, ensure_ascii=False), encoding='utf-8')
            self.assertEqual(md.read_corpus(path)[0]['draft'], '')
            path.write_text(json.dumps({'text': 'Строки стиха'}), encoding='utf-8')
            self.assertRaisesRegex(ValueError, 'PD', md.read_corpus, path)


if __name__ == '__main__':
    unittest.main()
