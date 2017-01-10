#!/usr/bin/python
# -*- coding: utf-8 -*-

# Attension Translator Model

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable,\
    optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import MeCab

s_word2id = []
t_word2id = []
s_id2word = []
t_id2word = []

def to_words(sentence):
    """
    入力: 'すべて自分のほうへ'
    出力: tuple(['すべて', '自分', 'の', 'ほう', 'へ'])
    """
    tagger = MeCab.Tagger('mecabrc')
    mecab_result = tagger.parse(sentence)
    info_of_words = mecab_result.split('\n')
    
    words = []
    for info in info_of_words:
        # macabで分けると、文の最後に’’が、その手前に'EOS'が来る
        if info == 'EOS' or info == '':
            break
            # info => 'な\t助詞,終助詞,*,*,*,*,な,ナ,ナ'
        info_elems = info.split(',')
        #print(info_elems)              # MeCab analysis result
        # 6番目に、無活用系の単語が入る。もし6番目が'*'だったら0番目を入れる
        if info_elems[6] == '*':
            # info_elems[0] => 'ヴァンロッサム\t名詞'
            words.append(info_elems[0][:-3])
            continue
        words.append(info_elems[6])
    #return tuple(words)
    return words
    
def load_data(fname):
    word2id = {}
    id2word = {}
    sentence = open(fname).read().split('\n')
    #print("open file=", fname)
    for i in range(len(sentence)):
        words = []
        #print("sentence[i]=", sentence[i])
        words = to_words(sentence[i])
        #print("words=", words)
        for word in words:
            if word not in word2id:
                #print("add word=", word)
                idx = len(word2id)
                id2word[idx] = word
                word2id[word] = idx
    idx = len(word2id)
    id2word[idx] = '<eos>'
    word2id['<eos>'] = idx
    numlines = len(sentence)
    numvocab = len(word2id)
    return numlines, numvocab, word2id, id2word, sentence

def mk_ct(gh, ht):
    alp = []
    s = 0.0
    for i in range(len(gh)):
        s += np.exp(ht.dot(gh[i]))  # exp(htとghの内積)の総和
    ct = np.zeros(100)
    for i in range(len(gh)):
        alpi = np.exp(ht.dot(gh[i]))/s  # exp(htとgh[i]の内積)
        ct += alpi * gh[i]          # context vector = alpha[i]とgh[i]の内積の総和
    ct = Variable(np.array([ct]).astype(np.float32))
    return ct

class MyATT(chainer.Chain):
    def __init__(self, s_numvocab, t_numvocab, dim):
        super(MyATT, self).__init__(
            embedx = L.EmbedID(s_numvocab, dim),
            embedy = L.EmbedID(t_numvocab, dim),
            H = L.LSTM(dim, dim),
            Wc1 = L.Linear(dim, dim),
            Wc2 = L.Linear(dim, dim),
            W = L.Linear(dim, t_numvocab),
        )
    def __call__(self, jline, eline):
        gh = []
        self.H.reset_state()
        for i in range(len(jline)):
            wid = s_word2id[jline[i]]
            x_dim = self.embedx(Variable(np.array([wid],dtype=np.int32)))
            h = self.H(x_dim)
            gh.append(np.copy(h.data[0]))   # append h to gh[]
        x_dim = self.embedx(\
            Variable(np.array([s_word2id['<eos>']],dtype=np.int32)))
        x_train = Variable(np.array([t_word2id[eline[0]]],dtype=np.int32))
        h = self.H(x_dim)
        ct = mk_ct(gh, h.data[0])
        h2 = F.tanh(self.Wc1(ct)+self.Wc2(h))
        accum_loss = F.softmax_cross_entropy(self.W(h2),x_train)
        #print ("accum_loss=", accum_loss)
        
        for i in range(len(eline)):
            wid = t_word2id[eline[i]]
            x_dim = self.embedy(
                Variable(np.array([wid],dtype=np.int32)))
            wid_next = t_word2id['<eos>'] \
                if (i==len(eline)-1) else t_word2id[eline[i+1]]
            x_train = Variable(np.array([wid_next],dtype=np.int32))
            h = self.H(x_dim)           # output of encoder part
            ct = mk_ct(gh, h.data)      # make context vector
            h2 = F.tanh(self.Wc1(ct)+self.Wc2(h))   # total output
            loss = F.softmax_cross_entropy(self.W(h2),x_train)
            accum_loss += loss
            #print ("accum_loss=", accum_loss.data)
        return accum_loss
    
    def reset_state(self):
        self.H.reset_state()

s_numlines, s_numvocab, s_word2id, s_id2word, s_sentence = load_data('talk_q.txt')
t_numlines, t_numvocab, t_word2id, t_id2word, t_sentence = load_data('talk_a.txt')

print("s_numlines=",s_numlines)
print("t_numlines=",t_numlines)
print("s_numvocab=",s_numvocab)
print("t_numvocab=",t_numvocab)
#print("s_word2id=",s_word2id)
#print("t_word2id=",t_word2id)

demb = 100
model = MyATT(s_numvocab,t_numvocab,demb)
optimizer = optimizers.Adam()
optimizer.setup(model)

jline = []
eline = []

for epoch in range(100):
    sum_loss = 0.0
    for i in range(len(s_sentence)-1):
    #for i in range(10):
        jline = to_words(s_sentence[i])
        #print ("divided jline=", jline)
        jliner = jline[::-1]
        eline = to_words(t_sentence[i])
        #print ("eline=", eline)
        model.reset_state()
        model.zerograds()
        loss = model(jliner, eline)
        loss.backward()
        loss.unchain_backward() # truncate
        optimizer.update()
        sum_loss += loss.data
        #print (i, " finished, loss=", loss)
    outfile = "./model/attention-" + str(epoch) + ".model"
    serializers.save_npz(outfile, model)
    print("epoch=", epoch, ", sum_loss=", sum_loss)

