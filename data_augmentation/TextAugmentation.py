import spacy
import sense2vec
import kenlm
from nltk.corpus import wordnet
import copy, random
#from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
from nltk.tag.perceptron import PerceptronTagger
import random


class TextAugmentation:
    
    def __init__(self, lm_path='/ihome/.../data/gigaword/lm.bin',
                        s2v_path='/ihome/.../data/reddit_vectors-1.1.0',
                        spacy_model='en_core_web_lg',
                        custom_vocab=[]):
        
        self._s2v = sense2vec.load(s2v_path)
        self._nlp = spacy.load(spacy_model)
        self._lm = kenlm.Model(lm_path)
        
        self.vocab = list(self._nlp.vocab.strings)
        
        if len(custom_vocab) == 0:
            self.custom_vocab = self.vocab
        else:
            self.custom_vocab = custom_vocab

    def is_ascii(self, s):
        return all(ord(c) < 128 for c in s)
    
    def get_most_similar_synonym(self, context, pos='ADJ'):
        #last word of the context
        word = context.split()[-1]
        
        q = u"{}|{}".format(word, pos)
        if q not in self._s2v:
            return ''

        _, v = self._s2v[q]
        syn, prob = self._s2v.most_similar(v, n=10)
        for s, p in list(zip(syn, prob)):
            if word.lower() not in s.lower():
                if s != None and self.is_ascii(s):
                    if s.endswith(pos):
                        return s.rstrip(pos).rstrip('|')

        return '' 
    
    def get_all_synonyms(self, context):
        #last word of the context
        word = context.split()[-1]

        lst = []
        for s in wordnet.synsets(word, pos=wordnet.ADJ):
            #we may remove this constrainet to create other kind of near-miss samples
            #if s.pos() != 'a':
            #    continue
            for l in s.lemmas(): 
                if l.name() not in lst:
                    lst.append(l.name())
            #lst.append('*******')
            for l in s.similar_tos(): 
                for l1 in l.lemmas(): 
                    if l1.name() not in lst:
                        lst.append(l1.name())
            #lst.append('*******')
            for l in s.also_sees():
                for l1 in l.lemmas(): 
                    if l1.name() not in lst:
                        lst.append(l1.name())
        if word in lst:
            lst.remove(word)
        return lst
    
    def get_most_likely_next_words(self, context, vocab=[]):

        if len(vocab)==0:
            vocab = self.vocab
            
        #_context_ already contains _word_
        # we compute the lm score up to the context

        state = kenlm.State()
        out = kenlm.State()
        self._lm.BeginSentenceWrite(state)
        sc = 0.0
        for w in context.split():
            sc += self._lm.BaseScore(state, w, out)
            state = copy.copy(out)

        # then we compute the lm score for each _w_ given _context_
        # _state_ embeded the _context_ so we are not modifing it to preserve the _context_

        candScore = {} 
        for w in vocab:
            x = w.strip()
            candScore[x] = self._lm.BaseScore(state, x, out)

        return sorted(candScore.items(), key=lambda kv: kv[1], reverse=True)

    def get_most_likely_next_adjective(self, context):
        '''
        mlnext = self.get_most_likely_next_words(context, vocab=self.custom_vocab)
        if len(mlnext) == 0:
            return []
        
        idx = random.randint(0, max(3, len(mlnext)-1))
        #increase the chance of most probable one
        if idx % 3 == 0:
            idx = 0
            
        return mlnext[idx][0]
        '''
        return self.get_most_likely_next_words(context, vocab=self.custom_vocab)[0][0]
    
    def get_most_likely_next_synonym(self, context):
        '''
        mlnext = self.get_most_likely_next_words(context, vocab=self.get_all_synonyms(context))
        if len(mlnext) == 0:
            return []
        
        idx = random.randint(0, max(3, len(mlnext)-1))
        #increase the chance of most probable one
        if idx % 3 == 0:
            idx = 0
            
        return mlnext[idx][0]                             
        '''
        return self.get_most_likely_next_words(context, vocab=self.get_all_synonyms(context))[0][0]

    def get_duplicate(self, context):
        return context.split()[-1]

    def augment(self, sentences, augmenter, verbose=True):
        
        augmented_sentences = list()
        original_sentences = list()
        
        pbar = tqdm(total=len(sentences))
                  
        for i, sent in enumerate(sentences):
            i += 1
            if verbose:
                if i % 50 == 0:
                    pbar.update(50)

            # extract all adjectives
            adj = []
            doc = self._nlp.tokenizer(sent)
            for d in self._nlp.tagger(doc):
                if not d.is_ascii:
                    adj = []
                    break

                if d.pos_ == 'ADJ':
                    adj.append(d.i)


            # pick a random adjective
            gen_sent = ''
            random.shuffle(adj)
            while adj:
                idx = adj.pop()
                d = doc[idx]

                badFlag = False
                gen_sent = ''
                for d in doc:
                    gen_sent += d.text_with_ws 
                    if d.i == idx:
                        #word = d.text
                        context = gen_sent
                        
                        # get alternative 
                        out = augmenter(context)

                        if len(out) == 0:
                            badFlag = True
                            break

                        if gen_sent.endswith(' '):
                            gen_sent += "{} ".format(out)
                        else:
                            gen_sent += " {}".format(out)

                if badFlag:
                    continue

                if gen_sent != '':
                    break

            if gen_sent == '' or badFlag:
                continue

            augmented_sentences.append(gen_sent)
            original_sentences.append(sent)
        
        return original_sentences, augmented_sentences

    def augment2(self, sentences, augmenter, verbose=True):
        
        augmented_sentences = list()
        original_sentences = list()
        
        pbar = tqdm(total=len(sentences))
        tagger = PerceptronTagger()

                  
        for i, sent in enumerate(sentences):
            i += 1
            if verbose:
                if i % 50 == 0:
                    pbar.update(50)

            # extract all adjectives
            adj = []
            ssplit = sent.split()
            pos = tagger.tag(ssplit)
            
            for i, p in enumerate(pos):
                pos = p[1]
                    
                if pos == 'JJ':
                    adj.append(i)
                    

            # pick a random adjective
            gen_sent = ''
            random.shuffle(adj)
            while adj:
                idx = adj.pop()
                d = ssplit[idx]

                badFlag = False
                gen_sent = ''
                for i, d in enumerate(ssplit):
                    gen_sent += d + ' '
                    if i == idx:
                        #word = d.text
                        context = gen_sent
                        
                        # get alternative 
                        out = augmenter(context)

                        if len(out) == 0:
                            badFlag = True
                            break

                        if gen_sent.endswith(' '):
                            gen_sent += "{} ".format(out)
                        else:
                            gen_sent += " {}".format(out)

                if badFlag:
                    continue

                if gen_sent != '':
                    break

            if gen_sent == '' or badFlag:
                continue

            augmented_sentences.append(gen_sent.strip())
            original_sentences.append(sent)
        
        return original_sentences, augmented_sentences

    
