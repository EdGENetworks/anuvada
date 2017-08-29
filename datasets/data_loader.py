"""
A generic module for dataset creation
"""
from spacy.en import English
from collections import Counter
import numpy as np
import os
import cPickle
import pandas as pd

nlp = English()

class CreateDataset():

    def generate_tokens(self, description):
        doc = nlp(unicode(description, 'utf-8'))
        return [x.lower_ for x in doc]

    def prepare_vocabulary(self, data):
        unique_id = Counter()
        for doc in data:
            sample_tokens = self.generate_tokens(doc)
            for token in sample_tokens:
                unique_id.update({token: 1})
        return unique_id

    def create_threshold(self, counter_dict):
        return Counter(el for el in counter_dict.elements() if counter_dict[el] > 5)

    def create_token2id_dict(self, token_list):
        return {v: k for k, v in enumerate(token_list)}

    def create_dataset(self, x, y, folder_path, max_doc_tokens):
        data_tokens = []
        len_x = len(x)
        vocab_full = self.prepare_vocabulary(x)
        vocab_threshold = self.create_threshold(vocab_full)
        token2id = self.create_token2id_dict(list(vocab_threshold))
        token2id['_UNK'] = len(token2id)
        token2id['_PAD'] = len(token2id) + 1
        id2token = {k: v for k, v in enumerate(token2id)}
        label2id = {v: k for k, v in enumerate(list(set(y)))}
        id2label = {k: v for k, v in enumerate(label2id)}
        labels = [label2id[item] for item in y]
        for doc in x:
            sample_tokens = self.generate_tokens(doc)
            data_tokens.append([token2id.get(y, token2id['_UNK']) for y in sample_tokens])
        df = pd.DataFrame({'data_tokens': data_tokens,
                           'labels': labels})
        df['doc_len'] = df['data_tokens'].apply(lambda x: len(x))
        df = df.sort_values('doc_len', ascending=False)
        # Padding with dummy _UNK token
        lengths_array = df.doc_len.values
        # max_len = max(lengths_array)
        pad_token = len(token2id) + 1
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
        np.save(os.path.join(folder_path, 'labels_encoded'), df['labels'].values)
        return data_tokens, labels

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
            return samples_encoded, labels_encoded, token2id, label2id, length_masks
        except:
            print 'No dataset exists in the specified path.'
            return None

