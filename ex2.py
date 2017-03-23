# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 17:46:32 2016

@author: aileenlin
"""

import nltk
from nltk.corpus import gutenberg
gutenberg.fileids()
emma = gutenberg.words("austen-emma.txt")

"""The Brown Corpus is a convenient resource for studying systematic 
differences between genres, a kind of linguistic inquiry known as stylistics. 
 compare genres in their usage of modal verbs. 
produce the counts for a particular genre. """

from nltk.corpus import brown
news_text = brown.words(categories = 'hobbies')
fdist = nltk.FreqDist(w.lower() for w in news_text)
modals = ['what', 'who', 'when', 'where', 'why']
for m in modals:
    print(m + ":", fdist[m], end = '  ')
    
from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch']
cfd = nltk.ConditionalFreqDist(
                                (lang, len(word))
                                for lang in languages
                                for word in udhr.words(lang+'-Latin1'))
cfd.plot(cumulative = False)

from nltk.corpus import inaugural
inaugural.fileids()
[fileid[:4] for fileid in inaugural.fileids()]

"""Conditional Frequency Distribution"""

cfd = nltk.ConditionalFreqDist(
           (target, fileid[:4])
           for fileid in inaugural.fileids()
           for w in inaugural.words(fileid)
           for target in ['america', 'citizen']
           if w.lower().startswith(target))
cfd.plot()

from nltk.corpus import PlaintextCorpusReader
corpus_root = '/Users/aileenlin/Documents/NLTK/dict'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
wordlists.fileids()
wordlists.words('abc.txt')

genre_word = [(genre, word) 
               for genre in ['news', 'romance']
               for word in brown.words(categories=genre)]:
cfd = nltk.ConditionalFreqDist(genre_word)
print(cfd['news'])

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
cfd = nltk.ConditionalFreqDist(
                               (genre, word)
                               for genre in ['news', 'romance']
                               for word in brown.words(categories = genre)
                               if word in days   
                               )
cfd['news'].most_common()
cfd['romance'].most_common()
cfd.plot(cumulative = False)

def generate_model(cfdist, word, num = 15):
    for i in range(num):
        print(word, end = ' ')
        word = cfdist[word].max()

text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)

names = nltk.corpus.names
male_names = names.words('male.txt')
female_names = names.words('female.txt')
[w for w in male_names if w in female_names]

cfd = nltk.ConditionalFreqDist(
                               (fileid, name[-1])
                               for fileid in names.fileids()
                               for name in names.words(fileid))
cfd.plot()

x = [(fileid, name[-1]) 
     for fileid in names.fileids()
   for name in names.words(fileid)]

"""pronuncing dictionary """
entries = nltk.corpus.cmudict.entries()
syllable = ['N', 'IH0', 'K', 'S']
[pron for word, pron in entries if pron[-4:] == syllable]

p3 = [(pron[0]+'-'+pron[2], word) 
      for (word, pron) in entries
      if pron[0] == 'P' and len(pron) == 3] 
cfd = nltk.ConditionalFreqDist(p3)
for template in sorted(cfd.conditions()):
    if len(cfd[template]) > 10:
        words = sorted(cfd[template])
        wordstring = ' '.join(words)
        print(template, wordstring[:70] + "...")

prondict = nltk.corpus.cmudict.dict()
text = ['natural', 'language', 'processing']
[ph for w in text for ph in prondict[w][0]]

from nltk.corpus import swadesh
swadesh.fileids()
swadesh.words('en')
fr2en = swadesh.entries(['fr', 'en'])
fr2en
translate = dict(fr2en)
translate['chien']
translate['jeter']

"""word net / hierarchy / layers"""
from nltk.corpus import wordnet as wn
wn.synsets('motorcar')

motorcar = wn.synset('car.n.01')
motorcar.hypernyms()
paths = motorcar.hypernym_paths()
[synset.name() for synset in paths[1]]

wn.synset('water.n.01').part_meronyms()
wn.synset('water.n.01').substance__meronyms()
wn.synset('water.n.01').member_meronyms()

wn.synset('harmony.n.02').substance_meronyms()
dir(wn.synset('harmony.n.02'))
wn.synsets('harmony')
wn.synset('harmony.n.03').lemmas()