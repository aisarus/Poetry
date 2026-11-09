import tempfile
import unittest
from pathlib import Path

import phonetics
import poetry_lab


class PhoneticsTests(unittest.TestCase):
    def test_stressed_rhyme_and_non_rhyme(self):
        lexicon = phonetics.load_lexicon()
        rhyme = phonetics.compare_rhyme(
            phonetics.analyze_line('На крыше дома репетировали май', lexicon),
            phonetics.analyze_line('Не прозевай!', lexicon))
        self.assertEqual(rhyme['status'], 'candidate')
        self.assertEqual(rhyme['sound_tails_approx'], ['аj', 'аj'])
        different = phonetics.compare_rhyme(
            phonetics.analyze_line('вода', lexicon), phonetics.analyze_line('погода', lexicon))
        self.assertEqual(different['status'], 'different')

    def test_unknown_is_not_guessed(self):
        row = phonetics.analyze_line('неведомая галактика', phonetics.load_lexicon())
        self.assertIn('?', row['stress_pattern'])
        self.assertEqual(row['unknown_stress'], ['неведомая', 'галактика'])

    def test_lexicon_override_and_explicit_accents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stresses.txt'
            path.write_text('за́мок\nзамо́к # другой смысл\n', encoding='utf-8')
            lex = phonetics.load_lexicon(path)
            self.assertEqual(phonetics.word_stress('замок', lex)[1], 1)
            self.assertEqual(phonetics.word_stress('за́мок', lex)[1:], (0, 'explicit'))
            self.assertEqual(phonetics.word_stress('ёлка', lex)[1:], (0, 'yo'))

    def test_report_marks_uncertainty(self):
        report = poetry_lab.analyze('Неизвестная галактика\nМай\nНа крыше\nНе прозевай')
        self.assertEqual(report['stanzas'][0]['phonetic_lines'][0]['unknown_stress'],
                         ['неизвестная', 'галактика'])
        self.assertEqual(report['stanzas'][0]['rhyme_hints'][0]['phonetic']['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
