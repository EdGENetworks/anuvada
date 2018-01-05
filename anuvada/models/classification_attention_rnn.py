"""
A classification model based on recurrent neural network with attention
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import codecs
from gensim.models import Word2Vec
import numpy as np

from fit_module_rnn import FitModuleRNN


class AttentionClassifier(FitModuleRNN):

    def __init__(self, vocab_size, embed_size, gru_hidden, n_classes, bidirectional=True, word2vec_path = None):

        super(AttentionClassifier, self).__init__()
        self.num_tokens = vocab_size
        self.embed_size = embed_size
        self.gru_hidden = gru_hidden
        self.bidirectional = bidirectional
        self.n_classes = n_classes
        # self.batch_size = batch_size
        self.lookup = nn.Embedding(vocab_size, embed_size)
        if word2vec_path:
            model = Word2Vec.load(word2vec_path)
            wv_matrix = model.wv.syn0
            # adding additional vectors for _UNK and _PAD
            wv_matrix = np.insert(wv_matrix,wv_matrix.shape[1],0,axis=0)
            wv_matrix = np.insert(wv_matrix,wv_matrix.shape[1],0,axis=0)
            try:
                self.lookup.weight.data.copy_(torch.from_numpy(wv_matrix))
            except:
                print 'Please check your Word2Vec model...'
        self.gru = nn.GRU(embed_size, gru_hidden, bidirectional=True)
        self.weight_attention = nn.Parameter(torch.Tensor(2 * gru_hidden, 2 * gru_hidden))
        self.bias_attention = nn.Parameter(torch.Tensor(2 * gru_hidden, 1))
        self.weight_projection = nn.Parameter(torch.Tensor(2 * gru_hidden, 1))
        self.attention_softmax = nn.Softmax()
        self.final_softmax = nn.Linear(2 * gru_hidden, n_classes)
        torch.nn.init.xavier_uniform(self.weight_attention.data)
        torch.nn.init.xavier_uniform(self.weight_projection.data)
        torch.nn.init.constant(self.bias_attention.data, 0.1)
#        self.weight_attention.data.uniform_(-0.1, 0.1)
#       self.weight_projection.data.uniform_(-0.1, 0.1)
#        self.bias_attention.data.uniform_(-0.1, 0.1)

    def batch_matmul_bias(self, seq, weight, bias, nonlinearity=''):
        s = None
        bias_dim = bias.size()
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], weight)
            _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0, 1)
            if nonlinearity == 'tanh':
                _s_bias = torch.tanh(_s_bias)
            _s_bias = _s_bias.unsqueeze(0)
            if s is None:
                s = _s_bias
            else:
                s = torch.cat((s, _s_bias), 0)
        return s.squeeze()

    def batch_matmul(self, seq, weight, nonlinearity=''):
        s = None
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], weight)
            if nonlinearity == 'tanh':
                _s = torch.tanh(_s)
            _s = _s.unsqueeze(0)
            if s is None:
                s = _s
            else:
                s = torch.cat((s, _s), 0)
        return s.squeeze()

    def attention_mul(self, rnn_outputs, att_weights):
        attn_vectors = None
        for i in range(rnn_outputs.size(0)):
            h_i = rnn_outputs[i]
            a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
            h_i = a_i * h_i
            h_i = h_i.unsqueeze(0)
            if attn_vectors is None:
                attn_vectors = h_i
            else:
                attn_vectors = torch.cat((attn_vectors, h_i), 0)
        return torch.sum(attn_vectors, 0)

    def forward(self, x, mask, initial_state):
        # print padded_sequence.size()
        # print initial_state.size()
        embedded = self.lookup(x)
        embedded = embedded.transpose(0,1)
        masked_sequence = pack_padded_sequence(embedded, mask)
        rnn_output, _ = self.gru(masked_sequence, initial_state)
        attention_squish = self.batch_matmul_bias(pad_packed_sequence(rnn_output)[0], self.weight_attention,
                                                  self.bias_attention, nonlinearity='tanh')
        attention = self.batch_matmul(attention_squish, self.weight_projection)
        attention_norm = self.attention_softmax(attention.transpose(1, 0))
        attention_vector = self.attention_mul(pad_packed_sequence(rnn_output)[0], attention_norm.transpose(1, 0))
        # print attention_vector.unsqueeze(0).size()
        linear_map = self.final_softmax(attention_vector.squeeze(0))
        return linear_map

    def get_attention(self, x, mask):
        # print padded_sequence.size()
        x = Variable(x).type(self.embedding_tensor)
        initial_state = self.init_hidden()
        # print initial_state.size()
        embedded = self.lookup(x)
        embedded = embedded.transpose(0,1)
        masked_sequence = pack_padded_sequence(embedded, mask)
        rnn_output, _ = self.gru(masked_sequence, initial_state)
        attention_squish = self.batch_matmul_bias(pad_packed_sequence(rnn_output)[0], self.weight_attention,
                                                  self.bias_attention, nonlinearity='tanh')
        attention = self.batch_matmul(attention_squish, self.weight_projection)
        attention_norm = self.attention_softmax(attention.transpose(1, 0))
        return attention_norm

    def visualize_attention(self, x, mask, id2token, filepath):
        x = Variable(x).type(self.embedding_tensor)
        initial_state = self.init_hidden()
        # print initial_state.size()
        embedded = self.lookup(x)
        embedded = embedded.transpose(0,1)
        masked_sequence = pack_padded_sequence(embedded, mask)
        rnn_output, _ = self.gru(masked_sequence, initial_state)
        attention_squish = self.batch_matmul_bias(pad_packed_sequence(rnn_output)[0], self.weight_attention,
                                                  self.bias_attention, nonlinearity='tanh')
        attention = self.batch_matmul(attention_squish, self.weight_projection)
        attention_norm = self.attention_softmax(attention.transpose(1, 0))
        attention_values = attention_norm[0,:].data.cpu().numpy()
        x = list(x[0,:].data.cpu().numpy())
        words = [id2token[y] for y in x]
        with codecs.open(filepath, "w",encoding="utf-8") as html_file:
            for word, alpha in zip(words, attention_values/ attention_values.max()):
                html_file.write('<font style="background: rgba(255, 255, 0, %f)">%s</font>\n' % (alpha, word))
        return None

    def init_hidden(self):
        return Variable(torch.zeros(2, self.batch_size, self.gru_hidden)).type(self.dtype)
