# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 21:42:52 2017

@author: aileenlin
"""
import nltk, re, pprint
from nltk import word_tokenize
from urllib import request
url = "http://www.gutenberg.org/files/2554/2554.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
type(raw)
len(raw)
raw[:75]

tokens = word_tokenize(raw)
type(tokens)
tokens[:10]
text = nltk.Text(tokens)
type(text)
text[1024:1062]
tokens[1024:1062]
text.collocations()
text.concordance('young')

"""Processing HTML file"""
url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = request.urlopen(url).read().decode('utf8')
html[:60]

from bs4 import BeautifulSoup
raw = BeautifulSoup(html).get_text()
tokens = word_tokenize(raw)
tokens
text = nltk.Text(tokens)
text

"""Processing RSS Feeds"""
import feedparser
llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
llog['feed']['title']

s = input("Enter some text:")
print("You typed", len(word_tokenize(s)), "words.")

couplet = '''Squirrel has a new job.
            I am happy for him.'''
print(couplet)


a = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]
b = [' ' * 2 * (7 - i) + 'very' * i for i in a]
for line in b:
    print(line)

path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
f = open(path, encoding = 'latin2')
a = ""
for line in f:
    line = line.strip()
    a = a+line
    print(line)
    
import re
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
[w for w in wordlist if re.search('..j..t..', w)]

re.search('^m*i*e*$','me')
a = r'\band\b'
print(a)
 
"""Regular expression"""   
raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
well without--Maybe it's always pepper that makes people hot-tempered,'"""    
    
re.split(r' ',raw)
re.split(r'[ \n\t]',raw)    
re.split(r'\s+', raw)
re.split(r'\W+',raw) 
  
[int(n) for n in re.findall('[0-9]{2,}', '2009-12-31')]

regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
english_udhr = nltk.corpus.udhr.words('English-Latin1')
re.findall(regexp, english_udhr[0])

rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
cv_word_pairs = [(cv, w) for w in rotokas_words
                    for cv in re.findall(r'[ptksvr][aeiou]', w)]
                        
cv_index = nltk.Index(cv_word_pairs)


text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
sents = nltk.sent_tokenize(text)
pprint.pprint(sents[79:89])

text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"

def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last: i+1])
            last = i + 1
    words.append(text[last:])
    return words


def evaluate(text, segs):
    words = segment(text, segs)
    text_size = len(words)
    lexicon_size = sum(len(w)+1  for w in set(words))
    return text_size + lexicon_size

evaluate(text, seg1)

from random import randint

def flip(segs, pos):
    return segs[:pos] + str(1-int(segs[pos])) + segs[pos+1:]

def flip_n(segs, n):
    for i in range(n):
        segs = flip(segs, randint(0, len(segs)-1))
    return segs

def anneal(text, segs, iterations, cooling_rate):
    temperature = float(len(segs))
    while temperature > 0.5:
        best_segs, best = segs, evaluate(text, segs)
        for i in range(iterations):
            guess = flip_n(segs, round(temperature))
            score = evaluate(text, guess)
            if score < best:
                best, best_segs = score, guess
        score, segs = best, best_segs
        temperature = temperature / cooling_rate
        print(evaluate(text, segs), segment(text, segs))
    print()
    return segs
    
anneal(text, seg1, 5000, 1.2)

import os
os.getcwd()
os.chdir('/Users/aileenlin/Documents/NLTK/output')