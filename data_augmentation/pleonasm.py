#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import collections


# In[3]:


tips_path = '.../data/yelp2016/tips'
spc_path = '.../data/spc/raw'



# # Helper Functions

# In[4]:


def read_file(file_path):
    content = []
    with open(file_path) as f:
        for l in f:
            if l != '\n':
                content.append(l.strip())
    return content


def write_file(content, file_path):
    with open(file_path, 'w') as f:
        for l in content:
            f.write(l + '\n')


def clean_and_rewrite(file_path):
    content = read_file(file_path)
    write_file(content, file_path)


def create_non_parallel(file_path1, file_path2):
    content1 = read_file(file_path1)
    content2 = read_file(file_path2)

    del content1[1::2]
    del content2[0::2]

    if len(content1) > len(content2):
        del content1[-1]
    elif len(content2) > len(content1):
        del content2[-1]

    fp1 = file_path1[:-4] + '_np' + file_path1[-4:]
    fp2 = file_path2[:-4] + '_np' + file_path2[-4:]

    write_file(content1, fp1)
    write_file(content2, fp2)


# In[5]:


label = {'concise': '1', 'verbose': '0'}


def create_train_label(concise, verbose):
    sentences = []
    labels = []
    for l in concise:
        sentences.append(l)
        labels.append(label['concise'])
    for l in verbose:
        sentences.append(l)
        labels.append(label['verbose'])

    return sentences, labels


def train_dev_split(data, split_ratio=.2):
    idx_train_data = np.random.choice(len(data),
                                      size=int(len(data) * (1 - split_ratio)),
                                      replace=False)
    idx_dev_data = np.setdiff1d(range(0, len(data)), idx_train_data)

    data_train = np.array(data)[idx_train_data]
    data_dev = np.array(data)[idx_dev_data]

    return data_train, data_dev


def read_multi_aligned_data(concise_path,
                            verbose_path,
                            concise2_path='',
                            mix_ration=.5,
                            np=False):
    
    concise = read_file(concise_path)
    verbose = read_file(verbose_path)

    if concise2_path != '':
        concise2 = read_file(concise2_path)
        ln = min(len(concise), len(concise2))
        ln = int(ln * mix_ration)
        concise = concise[:ln] + concise2[ln:int(ln / mix_ration) - 1]

    if np:
        ln = min(len(concise), len(verbose))
        ln = int(ln / 2)
        concise = concise[:ln]
        verbose = verbose[ln:]

    return concise, verbose


def create_train_dev_file(concise_path,
                          verbose_path,
                          target_file_prefix,
                          concise2_path='',
                          mix_ration=.5,
                          split_ratio=.2,
                          np=False):

    concise, verbose = read_multi_aligned_data(concise_path, verbose_path,
                                               concise2_path, mix_ration, np)

    train_concise, dev_concise = train_dev_split(concise, split_ratio)
    train_verbose, dev_verbose = train_dev_split(verbose, split_ratio)

    train_sent, train_lbl = create_train_label(train_concise, train_verbose)
    dev_sent, dev_lbl = create_train_label(dev_concise, dev_verbose)

    write_file(train_sent, target_file_prefix + 'train.text')
    write_file(train_lbl, target_file_prefix + 'train.labels')

    write_file(dev_sent, target_file_prefix + 'dev.text')
    write_file(dev_lbl, target_file_prefix + 'dev.labels')


# In[6]:


def extract_vocab(con, ver, file=False):
    if file:
        con = read_file(con)
        ver = read_file(ver)
        
    vocab = collections.Counter()
    for s in con:
        vocab.update(s.split())
    for s in ver:
        vocab.update(s.split())
    
    return list(vocab)

def write_vocab(vocab, file_path):
    with open(file_path, 'w') as f:
        for v in vocab:
            f.write(v+'\n')

def update_vocab(v_main, v_update):
    return v_main + list(set(v_update) - set(v_main))


# # Reading Tips/SPC

# In[7]:


df_spc = pd.read_csv(spc_path + '/spc_clean.csv')
df_tip = pd.read_csv(tips_path + '/tips_25.csv')


# ### Read Sentences

# In[8]:


sent_spc = df_spc['sentence'].values
sent_tip = df_tip['sentence'].values


# In[9]:


concise_spc = df_spc[df_spc['pleonastic'] == 0]['sentence'].values
verbose_spc = df_spc[df_spc['pleonastic'] == 1]['sentence'].values


# In[10]:


len(sent_spc), len(sent_tip)


# In[14]:


s= 'it has been excruciating painful today .'
df_spc[df_spc['sentence'] == s]


# ### Read Vocabulary

# In[6]:


vocab = []
with open(tips_path + '/vocab_clean') as f:
    for w in f:
        vocab.append(w.strip())


# In[7]:


vocab[:5], len(vocab)


# In[8]:


vocab_aj = []
with open(tips_path + '/vocab_clean_aj') as f:
    for w in f:
        vocab_aj.append(w.strip())


# In[9]:


vocab_aj[:5], len(vocab_aj)


# In[10]:




# # Augmentation

# ### Initiating Augmentation Models

# In[11]:


import random


# In[12]:


from TextAugmentation import TextAugmentation


# In[13]:


textAug = TextAugmentation(s2v_path='.../data/reddit_vectors-1.1.0/', 
                           lm_path='.../data/gigaword/lm.bin', 
                           custom_vocab=vocab_aj)


# ### Verbose

# In[18]:


org_ver, aug_ver = textAug.augment(sentences=sent_tip,augmenter=textAug.get_most_similar_synonym)


# In[19]:


aug = aug_ver
org = org_ver

session_id = random.randint(1,1000)

with open('{}/augmented/{}_verbose.txt'.format(tips_path, session_id), 'w') as f:
    for l in aug:
        f.write(l+'\n')
with open('{}/augmented/{}_concise.txt'.format(tips_path, session_id), 'w+') as f:
    for l in org:
        f.write(l+'\n')


# ### Near-Miss Concise (Syn -> LM)

# In[20]:


get_ipython().run_cell_magic('time', '', 'org_nncon, aug_nncon = textAug.augment(sentences=sent_tip,augmenter=textAug.get_most_likely_next_synonym)')


# In[21]:


aug = aug_nncon
org = org_nncon

session_id = random.randint(1,1000)

with open('{}/augmented/{}_nn_concise.txt'.format(tips_path, session_id), 'w') as f:
    for l in aug:
        f.write(l+'\n')
with open('{}/augmented/{}_concise.txt'.format(tips_path, session_id), 'w+') as f:
    for l in org:
        f.write(l+'\n')


# ### Near-Miss Concise (AJ -> LM)

# In[23]:


org_adj, aug_adj = textAug.augment(sentences=sent_tip,augmenter=textAug.get_most_likely_next_adjective)


# In[25]:


aug = aug_adj
org = org_adj

session_id = random.randint(1,1000)

with open('{}/augmented/{}_adj_concise.txt'.format(tips_path, session_id), 'w') as f:
    for l in aug:
        f.write(l+'\n')
with open('{}/augmented/{}_concise.txt'.format(tips_path, session_id), 'w+') as f:
    for l in org:
        f.write(l+'\n')


# ### Duplicate

# In[14]:


org_dup, aug_dup = textAug.augment(sentences=sent_tip,augmenter=textAug.get_duplicateplicate)


# In[16]:


aug = aug_dup
org = org_dup

session_id = random.randint(1,1000)

with open('{}/augmented/{}_dup_verbose.txt'.format(tips_path, session_id), 'w') as f:
    for l in aug:
        f.write(l+'\n')
with open('{}/augmented/{}_concise.txt'.format(tips_path, session_id), 'w+') as f:
    for l in org:
        f.write(l+'\n')


# ### Non-Parallel

# In[ ]:


create_non_parallel(tips_path + '/augmented/617_concise.txt',
                    tips_path + '/augmented/617_verbose.txt')

create_non_parallel(tips_path + '/augmented/711_concise.txt',
                    tips_path + '/augmented/711_dup_verbose.txt')

create_non_parallel(tips_path + '/augmented/752_adj_concise.txt',
                    tips_path + '/augmented/752_concise.txt')

create_non_parallel(tips_path + '/augmented/449_concise.txt',
                    tips_path + '/augmented/449_nn_concise.txt')


# ### Vocab

# In[49]:


vocab_spc = extract_vocab(concise_spc, verbose_spc)


# In[50]:


len(vocab_spc)


# In[91]:


vocab_617 = extract_vocab(tips_path + '/augmented/617_concise.txt', tips_path + '/augmented/617_verbose.txt', file=True)
vocab_711 = extract_vocab(tips_path + '/augmented/711_concise.txt', tips_path + '/augmented/711_dup_verbose.txt', file=True)
vocab_752 = extract_vocab(tips_path + '/augmented/752_adj_concise.txt', tips_path + '/augmented/752_concise.txt', file=True)
vocab_449 = extract_vocab(tips_path + '/augmented/449_concise.txt', tips_path + '/augmented/449_nn_concise.txt', file=True)


# In[92]:


len(set(vocab_spc) - set(vocab_617)), len(set(vocab_spc) - set(vocab_711)), len(set(vocab_spc) - set(vocab_752)), len(set(vocab_spc) - set(vocab_449))


# In[93]:


vocab_617_np = extract_vocab(tips_path + '/augmented/617_concise_np.txt', tips_path + '/augmented/617_verbose_np.txt', file=True)
vocab_711_np = extract_vocab(tips_path + '/augmented/711_concise_np.txt', tips_path + '/augmented/711_dup_verbose_np.txt', file=True)
vocab_752_np = extract_vocab(tips_path + '/augmented/752_adj_concise_np.txt', tips_path + '/augmented/752_concise_np.txt', file=True)
vocab_449_np = extract_vocab(tips_path + '/augmented/449_concise_np.txt', tips_path + '/augmented/449_nn_concise_np.txt', file=True)


# In[94]:


len(set(vocab_spc) - set(vocab_617_np)), len(set(vocab_spc) - set(vocab_711_np)), len(set(vocab_spc) - set(vocab_752_np)), len(set(vocab_spc) - set(vocab_449_np))


# In[100]:


len(vocab_617_np)


# In[103]:


x = update_vocab(vocab_617_np, vocab_spc)
len(x)


# In[105]:


write_vocab(update_vocab(vocab_617, vocab_spc), tips_path + '/augmented/617_vocab')
write_vocab(update_vocab(vocab_711, vocab_spc), tips_path + '/augmented/711_vocab')
write_vocab(update_vocab(vocab_752, vocab_spc), tips_path + '/augmented/752_vocab')
write_vocab(update_vocab(vocab_449, vocab_spc), tips_path + '/augmented/449_vocab')

write_vocab(update_vocab(vocab_617_np, vocab_spc), tips_path + '/augmented/617_np_vocab')
write_vocab(update_vocab(vocab_711_np, vocab_spc), tips_path + '/augmented/711_np_vocab')
write_vocab(update_vocab(vocab_752_np, vocab_spc), tips_path + '/augmented/752_np_vocab')
write_vocab(update_vocab(vocab_449_np, vocab_spc), tips_path + '/augmented/449_np_vocab')


# # ML-Ready Train/Test File

# In[120]:


create_train_dev_file(tips_path + '/augmented/617_concise.txt',
                      tips_path + '/augmented/617_verbose.txt',
                      tips_path + '/augmented/ml_ready/617')
create_train_dev_file(tips_path + '/augmented/711_concise.txt',
                      tips_path + '/augmented/711_dup_verbose.txt',
                      tips_path + '/augmented/ml_ready/711')

create_train_dev_file(tips_path + '/augmented/752_adj_concise.txt',
                      tips_path + '/augmented/752_concise.txt',
                      tips_path + '/augmented/ml_ready/752')
create_train_dev_file(tips_path + '/augmented/449_concise.txt',
                      tips_path + '/augmented/449_nn_concise.txt',
                      tips_path + '/augmented/ml_ready/449')

##-----------Non-Parallel-------------------

create_train_dev_file(tips_path + '/augmented/617_concise_np.txt',
                      tips_path + '/augmented/617_verbose_np.txt',
                      tips_path + '/augmented/ml_ready/617_np')
create_train_dev_file(tips_path + '/augmented/711_concise_np.txt',
                      tips_path + '/augmented/711_dup_verbose_np.txt',
                      tips_path + '/augmented/ml_ready/711_np')

create_train_dev_file(tips_path + '/augmented/752_adj_concise_np.txt',
                      tips_path + '/augmented/752_concise_np.txt',
                      tips_path + '/augmented/ml_ready/752_np')
create_train_dev_file(tips_path + '/augmented/449_concise_np.txt',
                      tips_path + '/augmented/449_nn_concise_np.txt',
                      tips_path + '/augmented/ml_ready/449_np')


# ### Mix n Match

# the pair of sentences in 449 and 752 cannot form a meaningful dataset
# 
# - 449 should be used in conjunction with 617 to form _near-miss concise (syn)_ vs. _verbose (syn)_
# - 752 should be used in conjunction with 617 to form _near-miss concise (adj)_ vs. _verbose (syn)_
# - 449 should combine with 617 to form _near-miss concise (syn) + concise (tips)_ vs. _verbose (syn)_
# - 752 should combine with 617 to form _near-miss concise (adj) + concise (tips)_ vs. _verbose (syn)_
# 
# it is hard to make the combined dataset non-parallel by keeping odd and even index of each side respectively since the indexes might not be in sync
# 
# - to create non-parallel, we will use the one half of one side and the other half of the other side, e.g., c[:500], v[500:]
# 

# #### near-miss concise  vs. verbose 

# In[31]:


# near-miss concise (SYN+LM) vs. verbose (sense2vec)
create_train_dev_file(tips_path + '/augmented/449_nn_concise.txt',
                      tips_path + '/augmented/617_verbose.txt',
                      tips_path + '/augmented/ml_ready/449-617/')

# near-miss concise (ADJ+LM) vs. verbose (sense2vec)
create_train_dev_file(tips_path + '/augmented/752_adj_concise.txt',
                      tips_path + '/augmented/617_verbose.txt',
                      tips_path + '/augmented/ml_ready/752-617/')


# In[32]:


# Non-Parallel

# near-miss concise (SYN+LM) vs. verbose (sense2vec)
create_train_dev_file(tips_path + '/augmented/pairs/449_nn_concise.txt',
                      tips_path + '/augmented/pairs/617_verbose.txt',
                      tips_path + '/augmented/ml_ready/449-617/np/',
                      np=True)

# near-miss concise (ADJ+LM) vs. verbose (sense2vec)
create_train_dev_file(tips_path + '/augmented/pairs/752_adj_concise.txt',
                      tips_path + '/augmented/pairs/617_verbose.txt',
                      tips_path + '/augmented/ml_ready/752-617/np/',
                      np=True)


# #### concise + near-miss concise  vs. verbose 

# In[36]:


#50%-50%

# near-miss concise (SYN+LM) vs. verbose (sense2vec)
create_train_dev_file(
    concise_path=tips_path + '/augmented/pairs/617_concise.txt',
    verbose_path=tips_path + '/augmented/pairs/617_verbose.txt',
    target_file_prefix=tips_path + '/augmented/ml_ready/449n617-617/',
    concise2_path=tips_path + '/augmented/pairs/449_nn_concise.txt')

# near-miss concise (ADJ+LM) vs. verbose (sense2vec)
create_train_dev_file(
    concise_path=tips_path + '/augmented/pairs/617_concise.txt',
    verbose_path=tips_path + '/augmented/pairs/617_verbose.txt',
    target_file_prefix=tips_path + '/augmented/ml_ready/752n617-617/',
    concise2_path=tips_path + '/augmented/pairs/752_adj_concise.txt')


# In[38]:


#50%-50% + Non-Parallel

# near-miss concise (SYN+LM) vs. verbose (sense2vec)
create_train_dev_file(
    concise_path=tips_path + '/augmented/pairs/617_concise.txt',
    verbose_path=tips_path + '/augmented/pairs/617_verbose.txt',
    target_file_prefix=tips_path + '/augmented/ml_ready/449n617-617/np/',
    concise2_path=tips_path + '/augmented/pairs/449_nn_concise.txt',
    np=True)

# near-miss concise (ADJ+LM) vs. verbose (sense2vec)
create_train_dev_file(
    concise_path=tips_path + '/augmented/pairs/617_concise.txt',
    verbose_path=tips_path + '/augmented/pairs/617_verbose.txt',
    target_file_prefix=tips_path + '/augmented/ml_ready/752n617-617/np/',
    concise2_path=tips_path + '/augmented/pairs/752_adj_concise.txt',
    np=True)


# #### vocabs

# In[55]:


vocab_spc = extract_vocab(spc_path + '/spc_concise.txt', spc_path + '/spc_verbose.txt', file=True)


# In[59]:


vocab_449_617 = extract_vocab(tips_path + '/augmented/ml_ready/449-617/train.text',
                              tips_path + '/augmented/ml_ready/449-617/dev.text',
                              file=True)
vocab_752_617 = extract_vocab(tips_path + '/augmented/ml_ready/752-617/train.text',
                              tips_path + '/augmented/ml_ready/752-617/dev.text',
                              file=True)


# In[61]:


vocab_449_617_np = extract_vocab(tips_path + '/augmented/ml_ready/449-617/np/train.text',
                              tips_path + '/augmented/ml_ready/449-617/np/dev.text',
                              file=True)
vocab_752_617_np = extract_vocab(tips_path + '/augmented/ml_ready/752-617/np/train.text',
                              tips_path + '/augmented/ml_ready/752-617/np/dev.text',
                              file=True)


# In[65]:


vocab_449_617_mix = extract_vocab(tips_path + '/augmented/ml_ready/449n617-617//train.text',
                              tips_path + '/augmented/ml_ready/449n617-617/dev.text',
                              file=True)
vocab_752_617_mix = extract_vocab(tips_path + '/augmented/ml_ready/752n617-617/train.text',
                              tips_path + '/augmented/ml_ready/752n617-617/dev.text',
                              file=True)


# In[66]:


vocab_449_617_mix_np = extract_vocab(tips_path + '/augmented/ml_ready/449n617-617/np/train.text',
                              tips_path + '/augmented/ml_ready/449n617-617/np/dev.text',
                              file=True)
vocab_752_617_mix_np = extract_vocab(tips_path + '/augmented/ml_ready/752n617-617/np/train.text',
                              tips_path + '/augmented/ml_ready/752n617-617/np/dev.text',
                              file=True)


# In[68]:


write_vocab(update_vocab(vocab_449_617, vocab_spc), tips_path + '/augmented/ml_ready/449-617/vocab')
write_vocab(update_vocab(vocab_449_617_np, vocab_spc), tips_path + '/augmented/ml_ready/449-617/np/vocab')

write_vocab(update_vocab(vocab_752_617, vocab_spc), tips_path + '/augmented/ml_ready/752-617/vocab')
write_vocab(update_vocab(vocab_752_617_np, vocab_spc), tips_path + '/augmented/ml_ready/752-617/np/vocab')

write_vocab(update_vocab(vocab_449_617_mix, vocab_spc), tips_path + '/augmented/ml_ready/449n617-617/vocab')
write_vocab(update_vocab(vocab_449_617_mix_np, vocab_spc), tips_path + '/augmented/ml_ready/449n617-617/np/vocab')

write_vocab(update_vocab(vocab_752_617_mix, vocab_spc), tips_path + '/augmented/ml_ready/752n617-617/vocab')
write_vocab(update_vocab(vocab_752_617_mix_np, vocab_spc), tips_path + '/augmented/ml_ready/752n617-617/np/vocab')


# ### Write SPC

# In[122]:


write_file(concise_spc, spc_path + '/spc_concise.txt')
write_file(verbose_spc, spc_path + '/spc_verbose.txt')


# In[123]:


spc_sent, spc_lbl = create_train_label(concise_spc, verbose_spc)


# In[126]:


write_file(spc_sent, spc_path + '/spc.test.text')
write_file(spc_lbl, spc_path + '/spc.test.labels')


# # TODO
# 
# - Random pick from the suggestion list
# - Randomly insert before or after

# In[ ]:




