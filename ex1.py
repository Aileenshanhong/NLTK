# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 18:34:16 2016

@author: aileenlin
"""



import nltk
nltk.download()
from nltk.book import *
text1
text2

""" searching the text """
text1.concordance("monstrous")

""" words with similar context """
text1.similar("monstrous")
text2.similar("monstrous")
""" context shared by two or more words """
text2.common_contexts(["monstrous", "very"])

import numpy

""" the location of a word in a text """
text4.dispersion_plot(["citizens","democracy","freedom","duties","America"])
len(set(text3))
sorted(set(text3)
text5.count("lol")/len(text5) * 100

def lexical_diversity(text):
    return len(set(text))/len(text)
    
def percentage(count, total):
    return 100*count/total
    
lexical_diversity(text3)
lexical_diversity(text5)
percentage(text4.count('a'), len(text4))

X = ['Monty', 'Python', 'and', 'the', 'Holy', 'Grail']
sorted(X)
len(set(X))
X.count('the')

"""Frequency distribution"""
fdist1 = FreqDist(text1)
print(fdist1)
fdist1.most_common(50)
fdist1['whale']
fdist1.plot(cumulative = True)

V = set(text1)
long_words = [w for w in V if len(w)>15]
sorted(long_words)

fdist5 = FreqDist(text5)
sorted(w for w in set(text5) if len(w)>7 and fdist5[w]>7)

text4.collocations()
text8.collocations()

sorted(w for w in set(text7) if '-' in w and 'index' in w)
sorted(wd for wd in set(text3) if wd.istitle() and len(wd)>10)
sorted(w for w in set(sent7) if not w.islower())
sorted(w for w in set(text2) if 'cie' in w or 'cei' in w)
len(set(word.lower() for word in text1 if word.isalpha()))

for w in ['a', 'b', 'c']:
    print(w)
    
nltk.chat.chatbots() 
3*['Month', 'Python']   

len(text2)
len(set(text2))
text2.dispersion_plot(['Elinor', 'Marianne', 'Edward', 'Willoughby'])
text5.collocations()
my_sent = ['My','sent']
" ".join(my_sent).split()
len(my_sent)+len(my_sent)
len(my_sent+my_sent)
sent1[2][2]
sent3.index('the')

sorted(w for w in set(text5) if w.startswith('b'))

text9.index('sunset')
text9[621:644]

sorted(w for w in set(sent1+sent2+sent3+sent4+sent5+sent6+sent7+sent8) if w.isalpha())
len(sorted(set(w.lower() for w in text1)))
len(sorted(w.lower() for w in set(text1)))

not "!".isupper()
text2[-2:]
x = set(w.lower() for w in text5 if w.isalpha() and len(w)==4)
y = FreqDist(x).most_common()

x = set(w for w in text6 if w.isupper())
for w in x:
    print(w)

x = [w for w in text6 if "pt" in w]
x = [w for w in text6 if w.istitle()]
 
sent = ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']
x = [w for w in sent if w.startswith("sh")]
x = [w for w in sent if len(w)>4]
sum(len(w) for w in sent) / len(sent)

def vocab_size(text):
    return len(set(w.lower() for w in text if w.isalpha()))

vocab_size(text1)

set(sent3) < set(text1)

def percent(word, text):
    return 100 * text.count(word) / len(text)
    
percent("storm", text1)




