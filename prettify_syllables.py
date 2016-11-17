#! /usr/bin/env python3
import re
import sys
import pandas as pd


def check_ru(x, axis):
        alphabet = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
        alphabet = alphabet + alphabet.lower() + 'и́о́а́ы́я́е́у́ю́э́- '
        for ch in list(x.lower()):
            if ch not in alphabet:
                return False
            return True


def remove_forbidden_chars(str_, axis=0):
    forbidden_chars = ('\u0301\u0300\u0340\u0341' +
                       '.!:;,?@#$%^&*œ∑´®†¥¨ˆøπ“‘åß©˙∆˚¬…æ«Ω≈çµ≤≥÷')
    forbidden_chars = forbidden_chars + ''.join(map(str, range(10)))
    return ''.join([ch for ch in str_ if (ch not in forbidden_chars)])


def normal_syllables(row, axis=1):
        tokens = re.split(r'\s*?\|\s*?', row[1])
        vowels = r'[аеёоуиэюяы]'
        for token in tokens:
            syllables = re.split('\s+', token)
            found_vowels = re.findall(vowels, row[0])
            if len(found_vowels) == len(syllables):
                return ' '.join(syllables)
        return None


def check_lengths(row, axis=1):
        syllables = re.split('\s+', row[1].strip())
        if len(row[0]) != len(''.join([s for s in syllables if s != ''])):
            return False
        return True


def prettify(filename, out_filename="normal_syllables.txt"):
    data = pd.read_table(filename, names=['word', 'syllables'])
    data = data.dropna()[data['word'].apply(check_ru, axis=1)]
    data['syllables'] = data['syllables'].apply(remove_forbidden_chars, axis=1)
    data['syllables'] = data.apply(normal_syllables, axis=1)
    data = data.dropna()
    data = data[data.apply(check_lengths, axis=1)]
    data.to_csv(out_filename, index=False, sep='\t')


if __name__ == '__main__':
    prettify(sys.argv[1], sys.argv[2])
