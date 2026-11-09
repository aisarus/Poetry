import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import poetry_lab as lab


class TestAnalysis(unittest.TestCase):
    def test_syllable_count(self):
        self.assertEqual(lab.syllables('Не прозевай!'), 4)
        self.assertEqual(lab.syllables('На крыше дома репетировали май'), 12)

    def test_short_line_and_scheme(self):
        poem = ('На крыше дома репетировали май\n'
                'И окна разливались акварелью\n'
                'Трамваи пахли свежей карамелью\n'
                'Не прозевай!')
        report = lab.analyze(poem)
        stanza = report['stanzas'][0]
        self.assertEqual(stanza['line_count'], 4)
        self.assertTrue(stanza['short_fourth_line_1_to_5_syllables'])
        self.assertEqual(stanza['expected_scheme'], 'ABBA')
        self.assertEqual(stanza['rhyme_hints'][0]['words'], ['май', 'прозевай'])

    def test_evaluation_only_no_ollama(self):
        with tempfile.TemporaryDirectory() as folder:
            text = Path(folder) / 'poem.txt'
            text.write_text('Море\nГоре\nВскоре\nБуря\n', encoding='utf-8')
            with patch.object(lab, 'ollama_chat', side_effect=AssertionError('No Ollama call expected')):
                result = lab.main(['--file', str(text), '--analyze-only', '--out', str(Path(folder) / 'out')])
            self.assertEqual(result, 0)
            self.assertEqual(len(list((Path(folder) / 'out').rglob('final.txt'))), 1)


if __name__ == '__main__':
    unittest.main()
