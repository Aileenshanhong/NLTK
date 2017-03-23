# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 06:36:20 2017

@author: aileenlin
"""

import nltk, re, pprint
from nltk import word_tokenize

raw = 'Red lorry, yellow lorry, red lorry, yellow lorry.'
text = word_tokenize(raw)
fdish = nltk.FreqDist(text)
sorted(fdish)
for key in fdish:
    print(key + ':', fdish[key], end = '; ')
    
fd = nltk.FreqDist(nltk.corpus.brown.words())
cumulative = 0.0
most_common_words = [word for (word, count) in fd.most_common()]

sent = ['Take', 'care', 'of', 'the', 'sense', ',', 'and', 'the',
        'sounds', 'will', 'take', 'care', 'of', 'themselves', '.']

def search1(substring, words):
    result = []
    for word in words:
        if substring in word:
            result.append(word)
    return result
    
def search2(substring, words):
    for word in words:
        if substring in word:
            yield word
            
lengths = list(map(len, nltk.corpus.brown.sents(categories='news')))

map()

[len([c for c in w if c.lower() in "aeiou"]) for w in sent]

"""create a nested dictionary structure"""
def insert(trie, key, value):
    if key:
        first, rest = key[0], key[1:]
        if first not in trie:
            trie[first] = {}
        insert(trie[first], rest, value)
    else:
        trie['value'] = value
        
        
trie = {}
insert(trie, 'chat', 'cat')
insert(trie, 'chien', 'dog')
insert(trie, 'chair', 'flesh')
insert(trie, 'apple', 'stylish')

trie = dict(trie)
pprint.pprint(trie, width = 40)

list(map(lambda w:len(list(filter(lambda c: c.lower() in "aeiou", w))), sent))

with open("readme.txt") as f:
    data = f.read()
    
    
def find_words(text, wordlength):  
    return list(filter(lambda w: len(w)==wordlength, text))



import pdb
find_words(['cat'], 3)

pdb.run("find_words(['cat'], 3)")

trie = {}
key = 'chat'
value = 'cat'

from numpy import arange
from matplotlib import pyplot

colors = 'rgbcmyk' # red, green, blue, cyan, magenta, yellow, black

def bar_chart(categories, words, counts):
    "Plot a bar chart showing counts for each word by category"
    ind = arange(len(words))
    width = 1 / (len(categories) + 1)
    bar_groups = []
    for c in range(len(categories)):
        bars = pyplot.bar(ind+c*width, counts[categories[c]], width,
                         color=colors[c % len(colors)])
        bar_groups.append(bars)
    pyplot.xticks(ind+width, words)
    pyplot.legend([b[0] for b in bar_groups], categories, loc='upper left')
    pyplot.ylabel('Frequency')
    pyplot.title('Frequency of Six Modal Verbs by Genre')
    pyplot.show()
    
genres = ['news', 'religion', 'hobbies', 'government', 'adventure']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfdist = nltk.ConditionalFreqDist(
              (genre, word)
              for genre in genres
              for word in nltk.corpus.brown.words(categories=genre)
              if word in modals)

counts = {}
for genre in genres:
     counts[genre] = [cfdist[genre][word] for word in modals]
bar_chart(genres, modals, counts)

from matplotlib import use, pyplot
use('Agg')
bar_chart(genres, modals, counts)
pyplot.savefig('modals.png') 
print('Content-Type: text/html')
print()
print('<html><body>')
print('<img src="modals.png"/>')
print('</body></html>')


import networkx as nx
import matplotlib
from nltk.corpus import wordnet as wn

"""a graph reflecting word net's structure"""
def traverse(graph, start, node):
    graph.depth[node.name] = node.shortest_path_distance(start)
    for child in node.hyponyms():
        graph.add_edge(node.name, child.name)
        traverse(graph, start, child)

def hyponym_graph(start):
    G = nx.Graph() 
    G.depth = {}
    traverse(G, start, start)
    return G

def graph_draw(graph):
    nx.draw_graphviz(graph,
         node_size = [16 * graph.degree(n) for n in graph],
         node_color = [graph.depth[n] for n in graph],
         with_labels = False)
    matplotlib.pyplot.show()
    
dog = wn.synset('dog.n.01')
graph = hyponym_graph(dog)
graph_draw(graph)