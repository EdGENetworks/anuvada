"""
A generic module for dataset creation
"""
import spacy
from collections import Counter
import numpy as np
import os
import cPickle
import pandas as pd
from tqdm import tqdm
from ..utils import to_multilabel
from gensim.models import Word2Vec

nlp = spacy.load('en')



class CreateDataset():

    def generate_tokens(self, description):
        doc = nlp(unicode(description))
        return [x.lower_ for x in doc]

    def prepare_vocabulary(self, data):
        unique_id = Counter()
        data_tokens = []
        for doc in tqdm(data):
            sample_tokens = self.generate_tokens(doc)
            data_tokens.append(sample_tokens)
            for token in sample_tokens:
                unique_id.update({token: 1})
        return unique_id, data_tokens

    def prepare_vocabulary_word2vec(self, data, folder_path):
        list_of_docs = []
        for doc in tqdm(data):
            doc = nlp(unicode(doc))
            list_of_sents = []
            for sentence in doc.sents:
                list_of_words = []
                for word in sentence:
                    list_of_words.append(word.orth_.lower())
                list_of_sents.append(list_of_words)
            list_of_docs.append(list_of_sents)
        flat_list = [item for sublist in list_of_docs for item in sublist]
        print 'Bulding Word2vec model...'
        model = Word2Vec(flat_list, size=300, window=5, min_count=5, workers=4)
        fname = 'word2vec_300_5_5'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        folder_fname = os.path.join(folder_path, fname)
        model.save(folder_fname)
        print 'Word2vec model saved...'
        unique_vocab = model.wv.index2word
        token2id = {v:k for k,v in enumerate(unique_vocab)}
        doc_tokens = []
        for doc in list_of_docs:
            temp = []
            for sent in doc:
                for word in sent:
                    temp.append(word)
            doc_tokens.append(temp)
        return token2id, doc_tokens

    def create_threshold(self, counter_dict):
        return Counter(el for el in counter_dict.elements() if counter_dict[el] >= 5)

    def create_token2id_dict(self, token_list):
        return {v: k for k, v in enumerate(token_list)}

    def create_dataset(self, x, y, folder_path, max_doc_tokens, multilabel=False, word2vec=False):
        len_x = len(x)
        if not word2vec:
            print 'Building vocabulary...'
            vocab_full, data_tokens = self.prepare_vocabulary(x)
            print 'Creating threshold...'
            vocab_threshold = self.create_threshold(vocab_full)
            token2id = self.create_token2id_dict(list(vocab_threshold))
            token2id['_UNK'] = len(token2id)
            token2id['_PAD'] = len(token2id)
            id2token = {k: v for k, v in enumerate(token2id)}
        else:
            print 'Building vocabulary...'
            token2id , data_tokens = self.prepare_vocabulary_word2vec(x, folder_path)
            token2id['_UNK'] = len(token2id)
            token2id['_PAD'] = len(token2id)
            id2token = {k: v for k, v in enumerate(token2id)}
        if multilabel is False:
            label2id = {v: k for k, v in enumerate(list(set(y)))}
            id2label = {k: v for k, v in enumerate(label2id)}
            label2count = pd.DataFrame(pd.Series(y).value_counts()).to_dict('dict')[0]
            labelid2count = {}
            for k, v in label2count.iteritems():
                labelid2count[label2id.get(k)] = v
            labels = [label2id[item] for item in y]
        else:
            labels_list = [xx.split('__') for xx in y]
            flat_labels = [item for sublist in labels_list for item in sublist]
            label2count = pd.DataFrame(pd.Series(flat_labels).value_counts()).to_dict('dict')[0]
            flat_labels = list(pd.Series(flat_labels).unique())
            label2id = {v: k for k, v in enumerate(flat_labels)}
            id2label = {k: v for k, v in enumerate(label2id)}
            labelid2count = {}
            for k, v in label2count.iteritems():
                labelid2count[label2id.get(k)] = v
            labels = [[label2id[y] for y in xx] for xx in labels_list]
        print 'Building dataset...'
        thresholded_tokens = []
        for document in data_tokens:
            thresholded_tokens.append([token2id.get(zz, token2id.get('_UNK')) for zz in document])
        df = pd.DataFrame({'data_tokens': thresholded_tokens, 'labels': labels})
        df['doc_len'] = df['data_tokens'].apply(lambda x: len(x))
        df = df.sort_values('doc_len', ascending=False)
        # Padding with dummy _UNK token
        lengths_array = df.doc_len.values
        # max_len = max(lengths_array)
        pad_token = token2id['_PAD']
        data_padded = np.zeros((len_x, max_doc_tokens), dtype=np.int)
        for i in xrange(data_padded.shape[0]):
            for j in xrange(data_padded.shape[1]):
                try:
                    data_padded[i][j] = df['data_tokens'].values[i][j]
                except IndexError:
                    data_padded[i][j] = pad_token

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(os.path.join(folder_path, 'samples_encoded'), data_padded)
        np.save(os.path.join(folder_path, 'lengths_mask'), df.doc_len.values)
        cPickle.dump(token2id, open(os.path.join(folder_path, 'token2id.pkl'), 'w'))
        cPickle.dump(label2id, open(os.path.join(folder_path, 'label2id.pkl'), 'w'))
        if multilabel:
            labels = df['labels'].values
            labels = [to_multilabel(y, len(label2id)) for y in labels]
            labels = np.array(labels)
            np.save(os.path.join(folder_path, 'labels_encoded'), labels)
            cPickle.dump(labelid2count, open(os.path.join(folder_path, 'labelid2count.pkl'), 'w'))

        else:
            np.save(os.path.join(folder_path, 'labels_encoded'), df['labels'].values)
            cPickle.dump(labelid2count, open(os.path.join(folder_path, 'labelid2count.pkl'), 'w'))
        print 'Datasets saved in folder %s' % (folder_path)
        return data_padded, labels, token2id, label2id, list(df.doc_len.values)

class LoadData():

    def __init__(self):
        return None

    def load_data_from_path(self, folder_path):
        try:
            samples_encoded = np.load(os.path.join(folder_path,'samples_encoded.npy'))
            labels_encoded = np.load(os.path.join(folder_path, 'labels_encoded.npy'))
            token2id = cPickle.load(open(os.path.join(folder_path, 'token2id.pkl'), 'r'))
            label2id = cPickle.load(open(os.path.join(folder_path, 'label2id.pkl'), 'r'))
            length_masks = np.load(os.path.join(folder_path, 'lengths_mask.npy'))
            labelid2count = cPickle.load(open(os.path.join(folder_path, 'labelid2count.pkl'), 'r'))
            return samples_encoded, labels_encoded, token2id, label2id, length_masks, labelid2count
        except:
            print 'No dataset exists in the specified path.'
            return None

