#!/usr/bin/env python
# coding: utf-8

# In[286]:


import pandas as pd
import numpy as np
import random


# ## Pos/Neg Vocabulary

# ### SST

# In[2]:


sst_folder_path = '.../data/sentiment/SST/'
data_folder_path = sst_folder_path + 'sentiment-analysis-on-movie-reviews/'
data_path = {
    'train': data_folder_path + '/train.tsv',
    'test': data_folder_path + '/test.tsv'
}


# In[3]:


df = pd.read_csv(data_path['train'], sep='\t')


# In[4]:


df = df.append(pd.read_csv(data_path['test'], sep='\t'))


# In[5]:


sentences = df.groupby(['SentenceId']).first()
len(sentences)


# In[6]:


#Positive
sst_4 = df[(df['Sentiment'] == 4) & (df['Phrase'].str.split().apply(len) == 1)]['Phrase'].str.lower().values
sst_3 = df[(df['Sentiment'] >= 3) & (df['Phrase'].str.split().apply(len) == 1)]['Phrase'].str.lower().values
#Negative
sst_0 = df[(df['Sentiment'] == 0) & (df['Phrase'].str.split().apply(len) == 1)]['Phrase'].str.lower().values
sst_1 = df[(df['Sentiment'] <= 1) & (df['Phrase'].str.split().apply(len) == 1)]['Phrase'].str.lower().values


# In[7]:


len(sst_4), len(sst_3), len(sst_0), len(sst_1)


# In[8]:


sst_4.tofile(sst_folder_path+'unigrams/positive-words-4.txt', sep='\n', format='%s')
sst_3.tofile(sst_folder_path+'unigrams/positive-words-3.txt', sep='\n', format='%s')
np.concatenate([sst_4, sst_3]).tofile(sst_folder_path+'unigrams/positive-words.txt', sep='\n', format='%s')

sst_0.tofile(sst_folder_path+'unigrams/negative-words-0.txt', sep='\n', format='%s')
sst_1.tofile(sst_folder_path+'unigrams/negative-words-1.txt', sep='\n', format='%s')
np.concatenate([sst_0, sst_1]).tofile(sst_folder_path+'unigrams/negative-words.txt', sep='\n', format='%s')


# In[9]:


sst_3 = pd.read_csv(sst_folder_path+'unigrams/positive-words-3.txt', sep='\n', header=None)[0].values
sst_1 = pd.read_csv(sst_folder_path+'unigrams/negative-words-1.txt', sep='\n', header=None)[0].values


# ### UIC

# In[10]:


uic_folder_path = '.../data/sentiment/UIC/'


# In[11]:


uic_p = pd.read_csv(uic_folder_path + 'positive-words.txt', header=None)[0].values
uic_n = pd.read_csv(uic_folder_path + 'negative-words.txt', header=None)[0].values
len(uic_p), len(uic_n)


# ### Merge

# In[12]:


len(set(sst_3) - set(uic_p)), len(set(uic_p) - set(sst_3))


# In[13]:


pos = list(set(sst_3).union(set(uic_p)))
len(pos)


# In[14]:


len(set(sst_1) - set(uic_n)), len(set(uic_n) - set(sst_1))


# In[15]:


neg = list(set(sst_1).union(set(uic_n)))
len(neg)


# In[16]:


vocab = dict()
vocab['positive'] = pos
vocab['negative'] = neg


# In[13]:


with open(sst_folder_path+'unigrams/positive-words-all.txt', 'w') as f:
    f.write('\n'.join(pos))


# In[12]:


with open(sst_folder_path+'unigrams/negative-words-all.txt', 'w') as f:
    f.write('\n'.join(neg))


# ### Read from Disk

# In[17]:


vocab = dict()
vocab['positive'] = pd.read_csv(sst_folder_path+'unigrams/positive-words-all.txt', header=None, sep='\n')[0].values
vocab['negative'] = pd.read_csv(sst_folder_path+'unigrams/negative-words-all.txt', header=None, sep='\n')[0].values


# In[18]:


len(vocab['positive']), len(vocab['negative'])


# ## Antonym Dictionary

# In[24]:


import nltk
from nltk.corpus import wordnet


# In[18]:


def WSD(context_sentence_str, ambiguous_word, pos=None, stem=True, hyperhypo=True, STOP=True):
    max_overlaps = 0; lesk_sense = None
    context_sentence = context_sentence_str.split()

    for ss in wordnet.synsets(ambiguous_word):
        # If POS is specified.
        if pos and ss.pos() is not pos:
            continue

        lesk_dictionary = []

        # Includes definition.
        lesk_dictionary += ss.definition().split()
        # Includes lemma_names.
        lesk_dictionary += ss.lemma_names()

        # Optional: includes lemma_names of hypernyms and hyponyms.
        if hyperhypo == True:
            #esk_dictionary+= list(chain(*[i.lemma_names() for i in deep_hypernyms_ss([ss])+deep_hypernyms_ss([ss])]))  
            lesk_dictionary+= list(chain(*[i.lemma_names() for i in ss.hypernyms()+ss.hyponyms()]))  

        if stem == True: # Matching exact words causes sparsity, so lets match stems.
            lesk_dictionary = [ps.stem(i) for i in lesk_dictionary]
            context_sentence = [ps.stem(i) for i in context_sentence] 

        overlaps = set()
        def_ss = ' '.join(lesk_dictionary)
        for w in context_sentence:
            if w in def_ss:
                overlaps.add(w)


        if STOP:
            overlaps = remove_stop(overlaps)
            context_sentence = remove_stop(context_sentence)

        embeddings = embed([' '.join(context_sentence), ' '.join(overlaps)])
        sim = np.inner(embeddings[0], embeddings[1])
   
        if sim > max_overlaps:
            lesk_sense = ss
            max_overlaps = sim
            
    return lesk_sense


# In[19]:


def get_antonyms(word):
    synset = wordnet.synsets(word)
    lems = sum(map(lambda s: s.lemmas(), synset), [])
    antonyms = sum(map(lambda l: l.antonyms(), lems), [])
    antonyms_names = list(set(map(lambda l: l.name(), antonyms)))
    return antonyms_names


# In[122]:


def get_synonyms(word):
    synset = wordnet.synsets(word)
    lems = sum(map(lambda s: s.lemmas(), synset), [])
    syn = set(map(lambda l: l.name(), lems)) - {word}
    return list(syn)


# In[20]:


antonym_dict = dict()
antonym_dict['positive'] = dict((filter(lambda kv: len(kv[1]) != 0, zip(vocab['positive'], list(map(get_antonyms, vocab['positive']))))))
antonym_dict['negative'] = dict((filter(lambda kv: len(kv[1]) != 0, zip(vocab['negative'], list(map(get_antonyms, vocab['negative']))))))


# In[21]:


len(vocab['positive']), len(antonym_dict['positive']), len(vocab['negative']), len(antonym_dict['negative'])


# In[22]:


pos_set = set(antonym_dict['positive'].keys())
neg_set = set(antonym_dict['negative'].keys())


# # Augmentation

# In[23]:


def H1(samples, antonym_dict, classes):
    samples_aug = {'positive': list(), 'negative': list()}
    for c in classes:
        for s in samples[c][:]:
            s_aug = []
            #print(s)
            for w in s.split():
                if w not in antonym_dict[c]:
                    s_aug.append(w)
                else:
                    s_aug.append(random.choice(antonym_dict[c][w]))
                    #print(f"{w} --> {antonym_dict[c][w]}")

            samples_aug[classes[c]].append(' '.join(s_aug))
            #print(' '.join(s_aug))
            #print()
    return samples_aug


# In[62]:


def get_candidate_s(sent, c):
    #tagged = nltk.pos_tag(nltk.word_tokenize(sent))
    tagged = nltk.pos_tag(sent)

    candidates = list(filter(lambda x: x[1] in ['JJ', 'NN'], tagged))
    
    #print(candidates)
    
    #cnd = random.choice(candidates)
    
    cnd = [(tagged.index(c), c) for c in candidates]
    
    return cnd


# In[40]:


def get_candidate(sent, c):
    #tagged = nltk.pos_tag(nltk.word_tokenize(sent))
    tagged = nltk.pos_tag(sent)
    tagged_dict = dict(tagged)

    if c == 'positive':
        intersections = pos_set.intersection(tagged_dict.keys())
    else:
        intersections = neg_set.intersection(tagged_dict.keys())

    jj_candidates = []
    candidates = []
    for w in intersections:
        pos = tagged_dict[w]
        if  pos == 'JJ':
            jj_candidates.append((w, pos)) 
        else:
            candidates.append((w, pos)) 
    
    if len(jj_candidates) != 0:
        candidates = jj_candidates
    
    #print(candidates)
    
    cnd = random.choice(candidates)
    
    return tagged.index(cnd), cnd[0]


# In[26]:


def H2(samples, antonym_dict, classes):
    samples_aug = {'positive': list(), 'negative': list()}
    for c in classes:
        for s in samples[c][:]:
            s_aug = []
            #print(s)
            s_tokenized = s.split()
            i, _ = get_candidate(s_tokenized, c)
            antonym = random.choice(antonym_dict[c][s_tokenized[i]])
            s_aug = s_tokenized[:i] + [antonym] + s_tokenized[i+1:]

            samples_aug[classes[c]].append(' '.join(s_aug))
            #print(' '.join(s_aug))
            #print()
    return samples_aug


# In[137]:


def H_SR(samples, classes):
    samples_aug = {'positive': list(), 'negative': list()}
    for c in classes:
        for s in samples[c][:]:
            s_aug = []
            #print(s)
            s_tokenized = s.split()
            cnd = get_candidate_s(s_tokenized, 'positive')
            random.shuffle(cnd)
            synonym = ''
            for x in cnd:
                synset = get_synonyms(s_tokenized[x[0]])
                #print(x, synset)
                if len(synset) > 0: 
                    #print(c, synset)
                    synonym = random.choice(synset)
                    s_aug = s_tokenized[:x[0]] + [synonym] + s_tokenized[x[0]+1:]
                    samples_aug[c].append(' '.join(s_aug))
                    break

       

    return samples_aug


# In[237]:


def H_SR2(samples, classes):
    samples_aug = {'positive': list(), 'negative': list()}
    for c in classes:
        for s in samples[c][:]:
            #print(s)
            s_tokenized = s.split()
            cnd = get_candidate_s(s_tokenized, 'positive')
            if len(cnd) == 0:
                continue
                
            cnd = random.choices(cnd, k=min(len(cnd), 3))
            synonym = ''
            for x in cnd:
                synset = get_synonyms(s_tokenized[x[0]])
                #print(x, synset)
                if len(synset) > 0: 
                    #print(c, synset)
                    synonym = random.choice(synset)
                    s_tokenized = s_tokenized[:x[0]] + [synonym] + s_tokenized[x[0]+1:]
            if s_tokenized == s.split():
                continue
                
            samples_aug[c].append(' '.join(s_tokenized))
            

    return samples_aug


# In[186]:


def H_RD(samples, classes):
    samples_aug = {'positive': list(), 'negative': list()}
    for c in classes:
        for s in samples[c][:]:
            s_aug = []
            s_tokenized = s.split()
            r = random.randint(0, len(s_tokenized))
            s_aug = s_tokenized[:r] + s_tokenized[r+1:]
            samples_aug[c].append(' '.join(s_aug))

    return samples_aug


# In[339]:


def H_RI(samples, classes, n=1):
    samples_aug = {'positive': list(), 'negative': list()}
    for c in classes:
        for s in samples[c][:]:
            s_aug = []
            #print(s)
            s_tokenized = s.split()
            cnd = get_candidate_s(s_tokenized, 'positive')
            random.shuffle(cnd)
            synonym = ''
            for x in cnd[:n]:
                synset = get_synonyms(s_tokenized[x[0]])
                if len(synset) > 0:
                    #print(c, synset)
                    synonym = random.choice(synset)
                    r = random.randint(0, len(s_tokenized))
                    s_aug = s_tokenized[:r] + [synonym] + s_tokenized[r:]
                    samples_aug[c].append(' '.join(s_aug))
                    break

    return samples_aug


# In[315]:


def H_RS(samples, classes, n=1):
    samples_aug = {'positive': list(), 'negative': list()}
    for c in classes:
        for s in samples[c][:]:
            s_tokenized = s.split()
            for i in range(0, n):
                r = random.sample(range(0, len(s_tokenized) - 1), 2)
                s_tokenized[r[0]], s_tokenized[r[1]] = s_tokenized[r[1]], s_tokenized[r[0]]

            samples_aug[c].append(' '.join(s_tokenized))

    return samples_aug


# In[204]:


def read_tf_file(text_file, label_file):
    df = {}
    try:
        with open(text_file) as tf, open(label_file) as lf:
            for t, l in zip(tf, lf):
                df[t.strip().strip("\"")] = int(l.strip())
    except OSError:
        pass
    
    return df


# In[205]:


def read_folder(file_prefx):
    #read tf files
    test_dict = dict()#read_tf_file(f"{file_prefx}.test.text", f"{file_prefx}.test.labels")
    train_dict = read_tf_file(f"{file_prefx}.train.text", f"{file_prefx}.train.labels")
    dev_dict = read_tf_file(f"{file_prefx}.dev.text", f"{file_prefx}.dev.labels")
    #convert to df
    test_df = pd.DataFrame.from_dict(test_dict, orient='index', columns=['label'])
    train_df = pd.DataFrame.from_dict(train_dict, orient='index', columns=['label'])
    dev_df = pd.DataFrame.from_dict(dev_dict, orient='index', columns=['label'])
    #merge
    all_df = pd.concat([test_df, train_df, dev_df]).reset_index()
    #extract setences
    sentences = {'positive': [], 'negative': []}
    sentences['positive'] = all_df[all_df['label'] == 1]['index'].values
    sentences['negative'] = all_df[all_df['label'] == 0]['index'].values
    return sentences


# In[206]:


def number_of_words(lst):
    return np.mean(list(map(lambda s: len(s.split()), lst)))


# In[207]:


classes = {'positive': 'negative', 'negative': 'positive'}


# In[208]:


def train_dev_split(samples, ratio=.9, cls=['positive', 'negative']):
    
    np.random.shuffle(samples[cls[0]])
    np.random.shuffle(samples[cls[1]])
    
    train_dev_cut_off_positive = int(len(samples[cls[0]])*ratio)
    train_dev_cut_off_negative = int(len(samples[cls[1]])*ratio)

    train = dict()
    train[cls[0]] = samples[cls[0]][:train_dev_cut_off_positive]
    train[cls[1]] = samples[cls[1]][:train_dev_cut_off_negative]

    dev = dict()
    dev[cls[0]] = samples[cls[0]][train_dev_cut_off_positive:]
    dev[cls[1]] = samples[cls[1]][train_dev_cut_off_negative:]
    
    return train, dev


# In[ ]:





# ## YPD Sample

# In[289]:


LOC = 'server'
#LOC = 'local'


# In[290]:


if LOC == 'local':
    ypd_path = '.../data/yelp2016/ypd/'
else:
    ypd_path = '.../framework/data/ypd/'


# In[291]:


ypd_sentences = read_folder(ypd_path + 'sentiment')


# In[ ]:





# ### Filter

# In[254]:


#filter for sentences with a positive/negative word that has an antonym
ypd_sentences['positive_filtered'] = list(filter(lambda s: len(set(s.split()).intersection(pos_set)) > 0, ypd_sentences['positive']))
ypd_sentences['negative_filtered'] = list(filter(lambda s: len(set(s.split()).intersection(neg_set)) > 0, ypd_sentences['negative']))


# ### Stat and Sample

# In[35]:


#number of sentence
print(f"# positive sentences:\t{len(ypd_sentences['positive'])}")
print(f"# negative sentences:\t{len(ypd_sentences['negative'])}")
pos_neg_ratio = len(ypd_sentences['positive']) / len(ypd_sentences['negative'])
print(f"ratio: {pos_neg_ratio}")
print(f"# positive filtered:\t{len(ypd_sentences['positive_filtered'])}")
print(f"# negative filtered:\t{len(ypd_sentences['negative_filtered'])}")


# In[36]:


print (f"mean words positive:\t\t{number_of_words(ypd_sentences['positive'])},\nmean words negative\t\t{number_of_words(ypd_sentences['negative'])}")
print (f"mean words positive-filtered:\t{number_of_words(ypd_sentences['positive_filtered'])},\nmean words negative-filtered\t{number_of_words(ypd_sentences['negative_filtered'])}")


# In[37]:


# number of antonym choices
np.mean(list(map(len, antonym_dict['positive'].values()))), np.mean(list(map(len, antonym_dict['negative'].values())))


# In[38]:


samples = dict()

samples['positive'] = np.random.choice(ypd_sentences['positive_filtered'], int(len(ypd_sentences['negative_filtered'])*pos_neg_ratio)).tolist()
samples['negative'] = ypd_sentences['negative_filtered']


# In[39]:


len(samples['positive']), len(samples['negative'])


# In[40]:


print (f"mean words positive samples:\t{number_of_words(samples['positive'])},\nmean words negative samples:\t{number_of_words(samples['negative'])}")


# ### Augment

# In[292]:


if LOC == 'local':
    ypd_aug_path = '.../data/sentiment/YPD/'
else:
    ypd_aug_path = '.../framework/data'


# In[293]:


samples_aug = {}


# #### H1

# In[43]:


samples_aug['H1'] = H1(samples, antonym_dict, classes)


# In[44]:


print("original:\t", len(samples['positive']), len(samples['negative']))
print("augmented:\t", len(samples_aug['H1']['positive']), len(samples_aug['H1']['negative']))


# ##### Non-parallel

# In[46]:


samples['positive_np']= samples['positive'][::2]
samples_aug['H1']['negative_np'] = samples_aug['H1']['negative'][1::2]

samples['negative_np']= samples['negative'][::2]
samples_aug['H1']['positive_np'] = samples_aug['H1']['positive'][1::2]


# In[50]:


print("\t\tPositive\tNegative")
print(f"original n-p:\t{len(samples['positive_np'])}\t\t{len(samples['negative_np'])}")
print(f"Augmented n-p:\t{len(samples_aug['H1']['positive_np'])}\t\t{len(samples_aug['H1']['negative_np'])}")


# In[48]:


list(zip(samples['positive'][:5], samples_aug['H1']['negative'][:5]))


# In[49]:


list(zip(samples['positive_np'][:5], samples_aug['H1']['negative_np'][:5]))


# ##### Combine

# In[52]:


samples_combined = dict()
samples_combined['positive'] = samples['positive_np'] + samples_aug['H1']['positive_np']
samples_combined['negative'] = samples['negative_np'] + samples_aug['H1']['negative_np']


# In[53]:


len(samples_combined['positive']), len(set(samples_combined['positive'])), len(samples_combined['negative']), len(set(samples_combined['negative']))


# ##### Combine parallel

# In[54]:


samples_combined['positive_pp'] = samples['positive'] + samples_aug['H1']['positive']
samples_combined['negative_pp'] = samples['negative'] + samples_aug['H1']['negative']


# In[55]:


len(samples_combined['positive_pp']), len(set(samples_combined['positive_pp'])), len(samples_combined['negative_pp']), len(set(samples_combined['negative_pp']))


# ##### Train/Dev Split PP

# In[57]:


train, dev = train_dev_split(samples_combined, cls=['positive_pp', 'negative_pp'])


# In[60]:


len(train['positive_pp']), len(dev['positive_pp']), len(train['negative_pp']), len(dev['negative_pp']),


# In[61]:


with open(ypd_aug_path+'H1/ypd.augmented.train.positive.ip', 'w') as f:
    f.write('\n'.join(train['positive_pp']))
with open(ypd_aug_path+'H1/ypd.augmented.train.negative.ip', 'w') as f:
    f.write('\n'.join(train['negative_pp']))


# In[62]:


with open(ypd_aug_path+'H1/ypd.augmented.dev.positive.ip', 'w') as f:
    f.write('\n'.join(dev['positive_pp']))
with open(ypd_aug_path+'H1/ypd.augmented.dev.negative.ip', 'w') as f:
    f.write('\n'.join(dev['negative_pp']))


# ##### Train/Dev Split

# In[81]:


train, dev = train_dev_split(samples_combined)


# In[82]:


len(train['positive']), len(dev['positive']), len(train['negative']), len(dev['negative']),


# In[369]:


with open(ypd_aug_path+'H1/ypd.augmented.train.positive', 'w') as f:
    f.write('\n'.join(train['positive']))
with open(ypd_aug_path+'H1/ypd.augmented.train.negative', 'w') as f:
    f.write('\n'.join(train['negative']))


# In[368]:


with open(ypd_aug_path+'H1/ypd.augmented.dev.positive', 'w') as f:
    f.write('\n'.join(dev['positive']))
with open(ypd_aug_path+'H1/ypd.augmented.dev.negative', 'w') as f:
    f.write('\n'.join(dev['negative']))


# #### H2

# In[63]:


samples_aug['H2'] = H2(samples, antonym_dict, classes)


# In[64]:


print("original:\t", len(samples['positive']), len(samples['negative']))
print("augmented:\t", len(samples_aug['H2']['positive']), len(samples_aug['H2']['negative']))


# ##### Non-Parallel

# In[65]:


samples['positive_np']= samples['positive'][::2]
samples_aug['H2']['negative_np'] = samples_aug['H2']['negative'][1::2]

samples['negative_np']= samples['negative'][::2]
samples_aug['H2']['positive_np'] = samples_aug['H2']['positive'][1::2]


# In[66]:


print("\t\tPositive\tNegative")
print(f"original n-p:\t{len(samples['positive_np'])}\t\t{len(samples['negative_np'])}")
print(f"original n-p:\t{len(samples_aug['H2']['positive_np'])}\t\t{len(samples_aug['H2']['negative_np'])}")


# In[67]:


list(zip(samples['positive'][:5], samples_aug['H2']['negative'][:5]))


# In[68]:


list(zip(samples['positive_np'][:5], samples_aug['H2']['negative_np'][:5]))


# ##### Combine

# In[69]:


samples_combined = dict()
samples_combined['positive'] = samples['positive_np'] + samples_aug['H2']['positive_np']
samples_combined['negative'] = samples['negative_np'] + samples_aug['H2']['negative_np']


# In[70]:


len(samples_combined['positive']), len(set(samples_combined['positive'])), len(samples_combined['negative']), len(set(samples_combined['negative']))


# ##### Combine parallel

# In[73]:


samples_combined['positive_pp'] = samples['positive'] + samples_aug['H2']['positive']
samples_combined['negative_pp'] = samples['negative'] + samples_aug['H2']['negative']


# In[74]:


len(samples_combined['positive_pp']), len(set(samples_combined['positive_pp'])), len(samples_combined['negative_pp']), len(set(samples_combined['negative_pp']))


# ##### Train/Dev Split

# In[196]:


train, dev = train_dev_split(samples_combined)


# In[197]:


len(train['positive']), len(dev['positive']), len(train['negative']), len(dev['negative']),


# In[199]:


with open(ypd_aug_path+'H2/data/ypd.augmented.train.positive', 'w') as f:
    f.write('\n'.join(train['positive']))
with open(ypd_aug_path+'H2/data/ypd.augmented.train.negative', 'w') as f:
    f.write('\n'.join(train['negative']))


# In[198]:


with open(ypd_aug_path+'H2/data/ypd.augmented.dev.positive', 'w') as f:
    f.write('\n'.join(dev['positive']))
with open(ypd_aug_path+'H2/data/ypd.augmented.dev.negative', 'w') as f:
    f.write('\n'.join(dev['negative']))


# ##### Train/Dev Split PP

# In[76]:


train, dev = train_dev_split(samples_combined, cls=['positive_pp', 'negative_pp'])


# In[77]:


len(train['positive_pp']), len(dev['positive_pp']), len(train['negative_pp']), len(dev['negative_pp']),


# In[78]:


with open(ypd_aug_path+'H2/ypd.augmented.train.positive.ip', 'w') as f:
    f.write('\n'.join(train['positive_pp']))
with open(ypd_aug_path+'H2/ypd.augmented.train.negative.ip', 'w') as f:
    f.write('\n'.join(train['negative_pp']))


# In[79]:


with open(ypd_aug_path+'H2/ypd.augmented.dev.positive.ip', 'w') as f:
    f.write('\n'.join(dev['positive_pp']))
with open(ypd_aug_path+'H2/ypd.augmented.dev.negative.ip', 'w') as f:
    f.write('\n'.join(dev['negative_pp']))


# #### SR

# In[294]:


ypd_sentences['positive_filtered'] = list(filter(lambda s: len(s.split()) > 5, ypd_sentences['positive']))
ypd_sentences['negative_filtered'] = list(filter(lambda s: len(s.split()) > 5, ypd_sentences['negative']))


# In[258]:


len(ypd_sentences['positive']), len(ypd_sentences['positive_filtered'])


# In[319]:


samples = dict()
samples['positive'] = ypd_sentences['positive_filtered'][:150000]
samples['negative'] = ypd_sentences['negative_filtered'][:150000]
len(samples['positive']), len(samples['negative'])


# In[241]:


samples_aug['SR2'] = H_SR2(samples, classes)


# In[150]:


samples_aug['SR'] = H_SR(samples, classes)


# In[242]:


print("original:\t", len(samples['positive']), len(samples['negative']))
print("augmented:\t", len(samples_aug['SR2']['positive']), len(samples_aug['SR2']['negative']))


# In[152]:


list(zip(samples['negative'][:5], samples_aug['SR']['positive'][:5]))


# ##### Train/Dev Split

# In[243]:


train, dev = train_dev_split(samples_aug['SR2'])


# In[244]:


len(train['positive']), len(dev['positive']), len(train['negative']), len(dev['negative']),


# In[156]:


with open(ypd_aug_path+'/SR/data/ypd.augmented.train.positive', 'w') as f:
    f.write('\n'.join(train['positive']))
with open(ypd_aug_path+'/SR/data/ypd.augmented.train.negative', 'w') as f:
    f.write('\n'.join(train['negative']))


# In[157]:


with open(ypd_aug_path+'/SR/data/ypd.augmented.dev.positive', 'w') as f:
    f.write('\n'.join(dev['positive']))
with open(ypd_aug_path+'/SR/data/ypd.augmented.dev.negative', 'w') as f:
    f.write('\n'.join(dev['negative']))


# In[249]:


with open(ypd_aug_path+'/SR2/data/ypd.augmented.train.text', 'w') as f:
    f.write('\n'.join(train['positive'][:50000]))
    f.write('\n')
    f.write('\n'.join(train['negative'][:50000]))
    f.write('\n')
        
with open(ypd_aug_path+'/SR2/data/ypd.augmented.train.labels', 'w') as f:
    for i in range(0, 50000):
        f.write('1\n')
    for i in range(0, 50000):
        f.write('0\n')


# #### RS

# In[294]:


ypd_sentences['positive_filtered'] = list(filter(lambda s: len(s.split()) > 5, ypd_sentences['positive']))
ypd_sentences['negative_filtered'] = list(filter(lambda s: len(s.split()) > 5, ypd_sentences['negative']))


# In[258]:


len(ypd_sentences['positive']), len(ypd_sentences['positive_filtered'])


# In[319]:


samples = dict()
samples['positive'] = ypd_sentences['positive_filtered'][:150000]
samples['negative'] = ypd_sentences['negative_filtered'][:150000]
len(samples['positive']), len(samples['negative'])


# In[ ]:


samples_aug['RS'] = H_RS(samples, classes, n=2)


# In[321]:


train, dev = train_dev_split(samples_aug['RS'])


# In[322]:


with open(ypd_aug_path+'/RS/data/ypd.augmented.train.text', 'w') as f:
    f.write('\n'.join(train['positive'][:50000]))
    f.write('\n')
    f.write('\n'.join(train['negative'][:50000]))
    f.write('\n')
        
with open(ypd_aug_path+'/RS/data/ypd.augmented.train.labels', 'w') as f:
    for i in range(0, 50000):
        f.write('1\n')
    for i in range(0, 50000):
        f.write('0\n')


# #### RI

# In[329]:


ypd_sentences['positive_filtered'] = list(filter(lambda s: len(s.split()) > 5, ypd_sentences['positive']))
ypd_sentences['negative_filtered'] = list(filter(lambda s: len(s.split()) > 5, ypd_sentences['negative']))


# In[330]:


len(ypd_sentences['positive']), len(ypd_sentences['positive_filtered'])


# In[347]:


samples = dict()
samples['positive'] = ypd_sentences['positive_filtered'][:150000]
samples['negative'] = ypd_sentences['negative_filtered'][:150000]
len(samples['positive']), len(samples['negative'])


# In[348]:


samples_aug['RI'] = H_RI(samples, classes, n=2)


# In[349]:


train, dev = train_dev_split(samples_aug['RI'])


# In[350]:


with open(ypd_aug_path+'/RI/data/ypd.augmented.train.text', 'w') as f:
    f.write('\n'.join(train['positive'][:50000]))
    f.write('\n')
    f.write('\n'.join(train['negative'][:50000]))
    f.write('\n')
        
with open(ypd_aug_path+'/RI/data/ypd.augmented.train.labels', 'w') as f:
    for i in range(0, 50000):
        f.write('1\n')
    for i in range(0, 50000):
        f.write('0\n')


# #### RD

# In[187]:


ypd_sentences['positive_filtered'] = list(filter(lambda s: len(s.split()) > 5, ypd_sentences['positive']))
ypd_sentences['negative_filtered'] = list(filter(lambda s: len(s.split()) > 5, ypd_sentences['negative']))


# In[188]:


len(ypd_sentences['positive']), len(ypd_sentences['positive_filtered'])


# In[194]:


samples = dict()
samples['positive'] = ypd_sentences['positive_filtered'][:150000]
samples['negative'] = ypd_sentences['negative_filtered'][:150000]
len(samples['positive']), len(samples['negative'])


# In[196]:


samples_aug['RD'] = H_RD(samples, classes)


# In[197]:


print("original:\t", len(samples['positive']), len(samples['negative']))
print("augmented:\t", len(samples_aug['RD']['positive']), len(samples_aug['RD']['negative']))


# In[198]:


list(zip(samples['positive'][:5], samples_aug['RD']['positive'][:5]))


# ##### Train/Dev Split

# In[199]:


train, dev = train_dev_split(samples_aug['RD'])


# In[200]:


len(train['positive']), len(dev['positive']), len(train['negative']), len(dev['negative']),


# In[201]:


with open(ypd_aug_path+'/RD/data/ypd.augmented.train.positive', 'w') as f:
    f.write('\n'.join(train['positive']))
with open(ypd_aug_path+'/RD/data/ypd.augmented.train.negative', 'w') as f:
    f.write('\n'.join(train['negative']))


# In[202]:


with open(ypd_aug_path+'/RD/data/ypd.augmented.dev.positive', 'w') as f:
    f.write('\n'.join(dev['positive']))
with open(ypd_aug_path+'/RD/data/ypd.augmented.dev.negative', 'w') as f:
    f.write('\n'.join(dev['negative']))


# In[ ]:





# ## SST

# In[86]:


sst_sentences = read_folder(sst_folder_path + 'ml/sst')


# In[87]:


#number of sentence
print(f"# positive sentences:\t{len(sst_sentences['positive'])}")
print(f"# negative sentences:\t{len(sst_sentences['negative'])}")
pos_neg_ratio = len(sst_sentences['positive']) / len(sst_sentences['negative'])
print(f"ratio: {pos_neg_ratio}")
print (f"mean words positive:\t\t{number_of_words(sst_sentences['positive'])},\nmean words negative\t\t{number_of_words(sst_sentences['negative'])}")


# ### Augment

# In[88]:


def get_all_candidates(tokens, c):
    #tagged = nltk.pos_tag(nltk.word_tokenize(sent))
    #tagged = nltk.pos_tag(sent)
    #tagged_dict = dict(tagged)
    #tagged_dict = dict(sent.split())

    if c == 'positive':
        intersections = pos_set.intersection(tokens)
    else:
        intersections = neg_set.intersection(tokens)

    return [(i, tokens.index(i)) for i in intersections]


# In[95]:


def H11(samples, antonym_dict=antonym_dict, classes=classes):
    samples_aug = {'positive': list(), 'negative': list()}
    for c in classes:
        for s in samples[c][:]:
            s_aug = []
            #print(s)
            s_tokenized = s.split()
            
            for t, i in get_all_candidates(s_tokenized, c):
                for a in antonym_dict[c][s_tokenized[i]]:
                    s_aug = s_tokenized[:i] + [a] + s_tokenized[i+1:]
                    samples_aug[classes[c]].append(' '.join(s_aug))


            #print(' '.join(s_aug))
            #print()
    return samples_aug


# In[91]:


sst_sentences['positive'][:5]


# In[96]:


sst_aug = dict()


# In[97]:


sst_aug['H11'] = H11(sst_sentences)


# In[101]:


len(sst_aug['H11']['positive']), len(sst_aug['H11']['negative'])


# In[ ]:




