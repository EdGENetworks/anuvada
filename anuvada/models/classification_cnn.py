"Text classification with CNN"

import torch
import torch.nn as nn
import torch.nn.functional as F
from fit_module_cnn import FitModuleCNN
from gensim.models import Word2Vec
import numpy as np


class ClassificationCNN(FitModuleCNN):

    def __init__(self, vocab_size, embed_dim, num_classes, kernel_num=256,
                 kernel_sizes=[3,4,5], dropout=0.5,word2vec_path=None):

        super(ClassificationCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.channel_size = 1
        self.dropout = nn.Dropout(dropout)

        self.embed = nn.Embedding(vocab_size, embed_dim)
        if word2vec_path:
            model = Word2Vec.load(word2vec_path)
            wv_matrix = model.wv.syn0
            # adding additional vectors for _UNK and _PAD
            wv_matrix = np.insert(wv_matrix,wv_matrix.shape[1],0,axis=0)
            wv_matrix = np.insert(wv_matrix,wv_matrix.shape[1],0,axis=0)
            self.embed.weight.data.copy_(torch.from_numpy(wv_matrix))
        # creating convolutional layers
        self.convs1 = nn.ModuleList([nn.Conv2d(self.channel_size, kernel_num, (K, embed_dim)) for K in kernel_sizes])
        self.final_linear = nn.Linear(len(kernel_sizes)*kernel_num, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        output = self.final_linear(x)
        return output


