# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 12:58:03 2017

@author: aileenlin
"""

""" tagging """

import nltk, re, pprint
from nltk import word_tokenize

text = word_tokenize("They refuse to permit us to obtain the refuse permit")
nltk.pos_tag(text)

text = nltk.Text(word.lower for word in nltk.corpus.brown.words())

sent = '''
       The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
       other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC
       Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPS
       said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/RB
       accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
       interest/NN of/IN both/ABX governments/NNS ''/'' ./.
      '''
[nltk.tag.str2tuple(t) for t in sent.split()]
nltk.corpus.brown.tagged_words(tagset = 'universal')

"""most common tags in brown corpus""""
from nltk.corpus import brown
brown_news_tagged = brown.tagged_words(categories = 'news', tagset = 'universal')
word_tag_pairs = nltk.bigrams(brown_news_tagged)
noun_preceders = [a[1] for (a, b) in word_tag_pairs if b[1] == 'NOUN']
fdist = nltk.FreqDist(noun_preceders)
[tag for (tag, _) in fdist.most_common()]

wsj = nltk.corpus.treebank.tagged_words(tagset = 'universal')
word_tag_fd = nltk.FreqDist(wsj)
[wt[0] for (wt,_) in word_tag_fd.most_common() if wt[1] == 'VERB']

cfd1 = nltk.ConditionalFreqDist(wsj)
cfd1['yield'].most_common()
cfd1['cut'].most_common()

tag_prefix = 'NN'
tagged_text = nltk.corpus.brown.tagged_words(categories = 'news')
cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                               if tag.startswith(tag_prefix))
dict((tag,cfd[tag].most_common(5)) for tag in cfd.conditions())   

from collections import defaultdict
alice = nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab = nltk.FreqDist(alice)
v1000 = [w for (w,_) in vocab.most_common(1000)]
mapping = defaultdict(lambda : 'UNK')
for v in v1000:
    mapping[v] = v

alice2 = list(map(lambda x: mapping[x], alice))                

from collections import defaultdict
counts = defaultdict(int)
from nltk.corpus import brown
for (word, tag) in brown.tagged_words(categories = 'news', tagset = 'universal'):
    counts[tag]+=1

counts['NOUN']

from operator import itemgetter
sorted(counts.items(), key = itemgetter(1), reverse=True)

words = nltk.corpus.words.words('en')
anagrams = defaultdict(list)
for word in words:
    key = ''.join(sorted(word))
    anagrams[key].append(word)
    
anagrams['aeilnrt']
nltk.Index((''.join(sorted(w)), w) for w in words)

from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories = 'news')
brown_sents = brown.sents(categories = 'news')

"""add patterns for tagging"""
patterns = [
     (r'.*ing$', 'VBG'),               # gerunds
     (r'.*ed$', 'VBD'),                # simple past
     (r'.*es$', 'VBZ'),                # 3rd singular present
     (r'.*ould$', 'MD'),               # modals
     (r'.*\'s$', 'NN$'),               # possessive nouns
     (r'.*s$', 'NNS'),                 # plural nouns
     (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
     (r'.*', 'NN')                     # nouns (default)
 ]
 
regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(brown_sents[3])
regexp_tagger.evaluate(brown_tagged_sents)
 
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.most_common(100)
likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags, backoff = nltk.DefaultTagger('NN'))
baseline_tagger.evaluate(brown_tagged_sents) 
sent = brown.sents(categories='news')[3]
baseline_tagger.tag(sent)


def performance(cfd, wordlist):
    lt = dict((word, cfd[word].max()) for word in wordlist)
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))

def display():
    import pylab
    word_freqs = nltk.FreqDist(brown.words(categories='news')).most_common()
    words_by_freq = [w for (w, _) in word_freqs]
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes = 2 ** pylab.arange(15)
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()
    
from nltk.corpus import brown
bown_tagged_sents = brown.tagged_sents(categories = 'news')
brown_sents = brown.sents(categories = 'news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
unigram_tagger.tag(brown_sents[2007])
unigram_tagger.evaluate(brown_tagged_sents)
