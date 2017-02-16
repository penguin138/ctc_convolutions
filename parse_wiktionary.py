#! /usr/bin/env python3
"""Basic parser for Wiktionary xml dump."""
from bs4 import BeautifulSoup
import re
import sys
import time


def parse_matches(matches):
    """Format strings with syllables."""
    parsed = []
    for match in matches:
        start_idx = match.find('слоги={{по-слогам|')
        match = match[start_idx + 18:-2]
        parsed.append(" ".join(match.split('|')))
    return parsed


def parse(filename):
    """Parse Russian Wiktionary dump."""
    with open(filename) as wiktionary:
        print("Reading xml...")
        start = time.time()
        soup = BeautifulSoup(wiktionary, "xml")
        end = time.time()
        print('Done in {} seconds'.format(end - start))
        print("Searching for pages...")
        pages = soup.findAll("page")
        print("Parsing pages...")
        with open("parsed_syllables.txt", 'w') as out_file:
            for page in pages:
                text = page.find("text")
                title = page.find("title")
                title_text = str(title.string)
                tag_text = str(text.string)
                vowels = re.findall(r'[аеиоуыэёюя]', title_text)
                print("Processing tag {}".format(str(title.string)))
                if re.search(r'язык=ru', tag_text):
                    if len(vowels) > 1:
                        matches = re.findall(r'слоги=\{\{по-слогам\|.*?}\}',
                                             tag_text)
                        parsed = parse_matches(matches)
                    else:
                        parsed = [title_text]
                    if len(parsed) >= 1:
                        out_file.write(title_text + "\t" + " | ".join(parsed) +
                                       '\n')


if __name__ == "__main__":
    parse(sys.argv[1])
