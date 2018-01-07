"Text classification with CNN"

import torch
import torch.nn as nn
import torch.nn.functional as F
from fit_module_cnn import FitModuleCNN
from gensim.models import Word2Vec
import numpy as np
from torch.autograd import Variable
import pandas as pd
import codecs


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
        self.embed.weight.requires_grad = True
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

    def compute_saliency_map(self, sample_doc, class_label, file_path, id2token):
        self.zero_grad()
        self.eval()
        sample_doc = Variable(sample_doc, requires_grad=False).type(self.embedding_tensor)
        sample_doc = sample_doc.unsqueeze(0)
        class_label = Variable(torch.LongTensor([class_label])).type(self.embedding_tensor)
        loss_function = torch.nn.NLLLoss(size_average=False)
        scores = self.forward(sample_doc)
        loss = loss_function(scores, class_label)
        #     print loss
        loss.backward()
        grad_of_param = {}
        for name, parameter in self.named_parameters():
            if 'embed' in name:
                grad_of_param[name] = parameter.grad
        grad_embed = grad_of_param['embed.weight']
        sensitivity = torch.pow(grad_embed, 2).mean(dim=1)
        sensitivity = list(sensitivity.data.cpu().numpy())
        jd = [id2token[zz] for zz in list(sample_doc.data.cpu().numpy()[0])]
        #     print list(sample_doc.data.cpu().numpy()[0])
        activations = [sensitivity[yy] for yy in list(sample_doc.data.cpu().numpy()[0])]
        df = pd.DataFrame({'word': jd, 'senstivity': activations})
        #     df = df.sort_values('senstivity',ascending=True)
        words = df.word.values
        values = df.senstivity.values
        with codecs.open(file_path, "w", encoding="utf-8") as html_file:
            for word, alpha in zip(words, values / values.max()):
                if not word == '_PAD':
                    html_file.write('<font style="background: rgba(255, 255, 0, %f)">%s</font>\n' % (alpha, word))
        return F.softmax(scores,dim=1).data.cpu().numpy(), df


