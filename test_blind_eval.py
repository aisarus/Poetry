import json
import tempfile
import unittest
from pathlib import Path

import blind_eval as ev


class BlindEvalTests(unittest.TestCase):
    def test_randomization_and_reveal_only_after_complete_ballot(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / 'pairs.jsonl', Path(tmp) / 'blind'
            rows = [{'topic_id': f't{i}', 'topic': f'Тема {i}',
                     'baseline': f'Базовый текст {i}', 'diffusion': f'Новый текст {i}'}
                    for i in range(6)]
            ev.write_jsonl(source, rows)
            self.assertEqual(ev.prepare(source, out, 7), 6)
            ballot, key = ev.read_jsonl(out / 'ballot.jsonl'), json.loads((out / 'key.json').read_text())
            self.assertRaises(ValueError, ev.score, out / 'ballot.jsonl', out / 'key.json')
            self.assertEqual({r['topic_id'] for r in ballot}, set(key))
            for row in ballot:
                self.assertEqual({row['A'], row['B']},
                                 {f'Базовый текст {row["topic_id"][1:]}', f'Новый текст {row["topic_id"][1:]}'})
                row['preferred'] = 'A' if key[row['topic_id']]['A'] == 'diffusion' else 'B'
            ev.write_jsonl(out / 'ballot.jsonl', ballot)
            result = ev.score(out / 'ballot.jsonl', out / 'key.json')
            self.assertEqual(result['diffusion_wins'], 6)
            self.assertEqual(result['baseline_wins'], 0)
            self.assertEqual(result['diffusion_share_of_decisive'], 1.0)
            self.assertRaises(FileExistsError, ev.prepare, source, out, 7)


if __name__ == '__main__':
    unittest.main()
