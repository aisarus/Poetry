import unittest

from wikisource_corpus import parse_poem, passages


class CorpusTests(unittest.TestCase):
    def test_extract_verse_only_without_line_numbers_or_style(self):
        html = ('<div class="headertemplate">Шапка</div><div class="poem"><p>'
                'Первая строка<br>Вторая строка<br><style>не текст</style>'
                '<span class="linenum">10 </span>Третья строка<br>Четвёртая строка'
                '</p></div><div>Примечания</div>')
        self.assertEqual(parse_poem(html),
                         'Первая строка\nВторая строка\nТретья строка\nЧетвёртая строка')

    def test_passages_keep_only_substantial_chunks(self):
        poem = '\n'.join(f'строка {n}' for n in range(25))
        self.assertEqual(len(passages(poem)), 1)
        self.assertEqual(len(passages(poem)[0].splitlines()), 20)


if __name__ == '__main__':
    unittest.main()
