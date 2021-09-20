#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from scipy import stats
from nltk import tokenize


# In[3]:


file_path = '.../specificity/ists/STSint.gs.headlines.wa'
new_file_path = '../../../data/more/news/'


# # iSTS 

# In[4]:


with open(file_path) as fp:
    soup = BeautifulSoup(fp, 'html.parser')


# In[5]:


items = list()
for c in soup.children:
    items.append(c)

#remove new lines
items = items[0::2]


# In[6]:


class relation:
    def __init__(self, type, score, phrases):
        self.type = type
        self.score = score
        self.phrases = phrases
    
    def __repr__(self):
        nl = '\n\t'
        return f"{self.type}\t// {self.score} // {' <==> '.join(self.phrases)}"
    def __str__(self):
        return self.__repr__()


# In[7]:


class specificity: 
    def __init__(self, sentences, relations):
        self.sentences = sentences
        self.relations = relations

    def __repr__(self):
        nl = '\n'
        return f"{nl.join(self.sentences)}\n{nl.join([str(r) for r in self.relations])}"
    def __str__(self):
        return self.__repr__()


# In[8]:


def parse_items(items):
    def extract_specificity(aln):
        return [i for i in aln.text.split('\n') if 'SPE' in i]     
    def parse_specificity(lst):
        rels = list()
        for s in lst:
            type = s.split('//')[1].strip()
            score = int(s.split('//')[2])
            phrases = [p.strip() for p in s.split('//')[3].split('<==>')]
            rels.append(relation(type, score, phrases))

        return rels

    spc_lst = list()
    for i in items:
        ch = list(i.children)
        sent = [s.strip('//').strip() for s in ch[0].splitlines()[1:]]
        #src = ch[2]
        #trs = ch[3]
        aln = ch[5]
        
        relations = parse_specificity(extract_specificity(aln))
        if len(relations) > 0:
            spc_lst.append(specificity(sent, relations))
    
    return spc_lst


# In[9]:


benchmark = parse_items(items)


# In[10]:


benchmark[0].relations


# In[11]:


from functools import reduce

print(reduce(lambda x, y: str(x)+'\n---------\n'+str(y), benchmark))


# In[12]:


lens = [len(s.split())  for b in benchmark for s in b.sentences]


# In[13]:


stats.describe(lens)


# # Headlines

# In[88]:


df1 = pd.read_csv(new_file_path + 'articles3.csv')


# In[89]:


titles = df1['title'].values
contents = df1['content'].values


# ## Titles

# In[91]:


titles = [str(t).split(' - ')[0].strip().lower() for t in titles if len(str(t).split())<15]
len(titles)


# In[49]:


titles = [' '.join(tokenize.word_tokenize(t)) for t in titles]


# In[ ]:


with open(new_file_path + 'titles.txt', 'a') as f:
    f.write('\n'.join(titles))


# ## Content

# In[93]:


sents =  [tokenize.sent_tokenize(c) for c in contents]


# In[94]:


sents = [s.strip().lower() for ss in sents for s in ss if len(s.split())<15]
len(sents)


# In[60]:


sents = [' '.join(tokenize.word_tokenize(s)) for s in sents]


# In[62]:


sents[1]


# In[ ]:


with open(new_file_path + 'sentences.txt', 'a') as f:
    f.write('\n'.join(sents))


# In[66]:


vocab_sents = [w for s in sents for w in s.split()]
vocab_sents = set(vocab_sents)
len(vocab_sents)


# In[67]:


with open(new_file_path + 'sentences.voc', 'w') as f:
    f.write('\n'.join(vocab_sents))


# In[71]:





# ## Read

# ### Titles

# In[14]:


titles = list()
with open(new_file_path + 'titles.txt', 'r') as f:
    titles = f.read().lower().splitlines()
len(titles)


# In[15]:


titles = [' '.join(tokenize.word_tokenize(t)) for t in titles]
titles = [t.strip().lower() for t in titles if len(t.split()) < 15]
len(titles)


# ### Content

# In[59]:


sents = list()
with open(new_file_path + 'sentences.txt', 'r') as f:
    sents = f.read().lower().splitlines()
len(sents)


# # Noisy Lables: Named Entities

# ## Titles

# In[100]:


import spacy
import matplotlib.pyplot as plt


# In[76]:


nlp = spacy.load('en_core_web_md')


# In[72]:


titles[5]


# In[78]:


doc = nlp(titles[5])


# In[90]:


ent = dict()
for t in titles:
    doc = nlp(t)
    ent[t] = doc.ents


# In[92]:


n_cnt = list()
s_cnt = 0
for t in titles:
    if len(ent[t]) > 0:
        s_cnt += 1
        n_cnt.append(len(ent[t]))


# In[171]:


len(titles), s_cnt, stats.describe(n_cnt)


# In[101]:


plt.hist(n_cnt)


# In[170]:


g0 = list()
s1 = list()
s2 = list()
s3p = list()

g0 = [e for e in ent if len(ent[e]) == 0]
s1 = [e for e in ent if len(ent[e]) == 1]
s2 = [e for e in ent if len(ent[e]) == 2]
s3p = [e for e in ent if len(ent[e]) > 2]


# In[169]:


len(g0), len(s1), len(s2), len(s3p)


# In[122]:


l_g = int(len(g0)/10)
l_s = int(len(s2)/10)
with open(new_file_path + '/ml-ready/no/specificity.dev.text', 'w') as f:
    f.write('\n'.join(g0[:l_g]))
    f.write('\n'.join(s2[:l_s]))
    f.write('\n')
with open(new_file_path + '/ml-ready/no/specificity.dev.labels', 'w') as f:
    f.write('\n'.join(['0'] * l_g))
    f.write('\n'.join(['1'] * l_s))
    f.write('\n')

with open(new_file_path + '/ml-ready/no/specificity.train.text', 'w') as f:
    f.write('\n'.join(g0[l_g:]))
    f.write('\n'.join(s2[l_s:]))
    f.write('\n')
with open(new_file_path + '/ml-ready/no/specificity.train.labels', 'w') as f:
    f.write('\n'.join(['0'] * (len(g0)-l_g)))
    f.write('\n'.join(['1'] * (len(s2)-l_s)))
    f.write('\n')


# ## Benchmark
# 
# 

# In[16]:


b_gen = list()
b_spec = list()
b_gen_phrase = list()
b_spec_phrase = list()

for b in benchmark:
    if 'SPE1' in b.relations[0].type:
        b_spec.append(b.sentences[0])
        b_gen.append(b.sentences[1])
        b_spec_phrase.append(b.relations[0].phrases[0])
        b_gen_phrase.append(b.relations[0].phrases[1])
    if 'SPE2' in b.relations[0].type:
        b_spec.append(b.sentences[1])
        b_gen.append(b.sentences[0])
        b_spec_phrase.append(b.relations[0].phrases[1])
        b_gen_phrase.append(b.relations[0].phrases[0])


# In[17]:


b_gen = [' '.join(tokenize.word_tokenize(t)) for t in b_gen]
b_gen = [t.strip().lower() for t in b_gen if len(t.split()) < 20]

b_spec = [' '.join(tokenize.word_tokenize(t)) for t in b_spec]
b_spec = [t.strip().lower() for t in b_spec if len(t.split()) < 20]


# In[25]:


l_gen = [len(t.split()) for t in b_gen]
l_spec = [len(t.split()) for t in b_spec]


# In[26]:


stats.describe(l_gen), stats.describe(l_spec)


# In[19]:


len(b_gen), len(b_spec)


# In[143]:


with open(new_file_path + '/ml-ready/no/specificity.test.text', 'w') as f:
    f.write('\n'.join(b_gen))
    f.write('\n'.join(b_spec))
    f.write('\n')
with open(new_file_path + '/ml-ready/no/specificity.test.labels', 'w') as f:
    f.write('\n'.join(['0'] * len(b_gen)))
    f.write('\n'.join(['1'] * len(b_spec)))
    f.write('\n')


# ## Augments

# In[139]:


from nltk.corpus import wordnet


# In[151]:


list(gold)[:5]


# In[156]:


wordnet.synsets('death_camp',pos=wordnet.NOUN)[0].hypernym_paths()[-1]


# In[172]:


gen_dev = list()
with open(new_file_path + '/ml-ready/no/specificity.dev.text', 'r') as f:
    gen_dev = f.readlines()[:100]


# In[174]:


gen_dev[:5]


# In[176]:


wordnet.synsets('graduate_school',pos=wordnet.NOUN)[0].hyponyms()


# In[ ]:


def get_ent(word):
    return list(
        map(lambda x: x.lemmas()[0].name(),
            wordnet.synsets(word,
                            pos=wordnet.NOUN)[0].hypernym_paths()[-1]))[-4:]


# # Vocab 

# In[ ]:


vocab_titles = [w for t in titles for w in t.split()]
vocab_titles = set(vocab_titles)
len(vocab_titles)


# In[ ]:


with open(new_file_path + 'titles.voc', 'w') as f:
    f.write('\n'.join(vocab_titles))


# In[ ]:





# In[ ]:


vocab_benchmark = [w.lower() for b in benchmark for s in b.sentences for w in tokenize.word_tokenize(s)]
vocab_benchmark = set(vocab_benchmark)
len(vocab_benchmark)


# In[ ]:


with open(new_file_path + 'benchmark.voc', 'w') as f:
    f.write('\n'.join(vocab_benchmark))


# In[ ]:





# In[ ]:


vocab_all = vocab_benchmark.union(vocab_titles).unionn(vocab_sents)
len(vocab_all)


# In[ ]:


with open(new_file_path + 'vocab', 'w') as f:
    f.write('\n'.join(vocab_all))


# # Evaluation

# ## Gold

# In[14]:


def parse_gold(benchmark):
    gold = dict()
    #gold2 = dict()
    for b in benchmark:
        for r in b.relations:
            if 'SPE1' in r.type:
                x = b.sentences[0]
                y = set(tokenize.word_tokenize(r.phrases[0].lower())) - set(tokenize.word_tokenize(r.phrases[1].lower()))
                #x2 = b.sentences[1]
                #y2 = set(tokenize.word_tokenize(r.phrases[1].lower())) - set(tokenize.word_tokenize(r.phrases[0].lower()))
            if 'SPE2' in r.type:
                x = b.sentences[1]
                y = set(tokenize.word_tokenize(r.phrases[1].lower())) - set(tokenize.word_tokenize(r.phrases[0].lower()))
                #x2 = b.sentences[0]
                #y2 = set(tokenize.word_tokenize(r.phrases[0].lower())) - set(tokenize.word_tokenize(r.phrases[1].lower()))

        x = ' '.join(tokenize.word_tokenize(x.lower()))
        gold[x] = y

        #x2 = ' '.join(tokenize.word_tokenize(x2.lower()))
        #gold2[x2] = y2
    
    return gold


# In[15]:


gold = parse_gold(benchmark)
len(gold)


