"""Integration-style smoke tests with a simulated local Ollama endpoint.

No external packages or GPU required. These tests verify orchestration and
artifacts, NOT the literary quality of a model's output.
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import poetry_lab as lab

DRAFT = (
    'На крыше дома репетировали май\n'
    'И окна разливались акварелью\n'
    'Трамваи пахли свежей карамелью\n'
    'Не прозевай!'
)
REVISED = (
    'На крыше дома репетировали май\n'
    'И окна разливались акварелью\n'
    'Трамваи пахли свежей карамелью\n'
    'Не опоздай!'
)


class TestFullRun(unittest.TestCase):
    def test_mocked_draft_critic_revision_creates_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(lab, 'ollama_chat', side_effect=[DRAFT, 'Рифма есть; ритм требует проверки.', REVISED]) as mocked:
                with contextlib.redirect_stdout(io.StringIO()):
                    status = lab.main(['--topic', 'Праздник', '--stanzas', '1', '--rounds', '1',
                                       '--out', tmp])
            self.assertEqual(status, 0)
            self.assertEqual(mocked.call_count, 3)  # draft + critique + revision
            dirs = [d for d in Path(tmp).iterdir() if d.is_dir()]
            self.assertEqual(len(dirs), 1)
            out = dirs[0]
            for filename in ['poem_00.txt', 'metrics_00.json', 'critique_00.txt',
                             'poem_01.txt', 'metrics_01.json', 'trace.json', 'final.txt']:
                self.assertTrue((out / filename).exists(), filename)
            self.assertEqual((out / 'final.txt').read_text(encoding='utf-8').strip(), REVISED)
            trace = json.loads((out / 'trace.json').read_text(encoding='utf-8'))
            self.assertEqual(len(trace), 2)
            self.assertIn('critique', trace[0])
            self.assertNotIn('critique', trace[1])
            self.assertEqual(trace[0]['prosody_report']['stanza_count'], 1)

    def test_draft_failure_has_nonzero_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(lab, 'ollama_chat', side_effect=RuntimeError('Ollama offline')):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    status = lab.main(['--rounds', '0', '--out', tmp])
            self.assertEqual(status, 1)

    def test_only_analysis_skips_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.txt'
            path.write_text(DRAFT, encoding='utf-8')
            with patch.object(lab, 'ollama_chat', side_effect=AssertionError('must not use model')):
                with contextlib.redirect_stdout(io.StringIO()):
                    status = lab.main(['--file', str(path), '--analyze-only', '--out', tmp])
            self.assertEqual(status, 0)


if __name__ == '__main__':
    unittest.main()
