from torch import nn
import torch
import logging
import torch.nn.functional as F
import numpy as np
import os
from models.attentions import Attention
from models.encoders import SentEncoder
from utils.model_utils import use_cuda, device
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# word2vec_path = os.path.join(current_path, 'data/emb/industry_vec.txt')
dropout = 0.15

class WordCNNEncoder(nn.Module):
    """[word cnn 编码， seq_len,word_dim ==> seq_len-filter_size+1 , out_channel == > out_channel,1]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, vocab, emb_vocab):
        super(WordCNNEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.word_dims = emb_vocab.word_dim  # 词向量的长度是 100 维

        # padding_idx 表示当取第 0 个词时，向量全为 0
        # 这个 Embedding 层是可学习的
        self.word_embed = nn.Embedding(
            vocab.word_size, self.word_dims, padding_idx=0)

        # pretrained_embs
        # extword_embed = vocab.load_pretrained_embs(word2vec_path)
        extword_embed = emb_vocab.embeddings
        extword_size, word_dims = extword_embed.shape
        logging.info("Load extword embed: words %d, dims %d." %
                     (extword_size, word_dims))

        # # 这个 Embedding 层是不可学习的  <=== requires_grad = False
        self.extword_embed = nn.Embedding(
            extword_size, word_dims, padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
        self.extword_embed.weight.requires_grad = False  # 不对这个变量求梯度

        input_size = self.word_dims

        self.filter_sizes = [2, 3, 4]  # n-gram window
        self.out_channel = emb_vocab.word_dim
        # 3 个卷积层，卷积核大小分别为 [2,100], [3,100], [4,100]
        self.convs = nn.ModuleList([nn.Conv2d(1, self.out_channel, (filter_size, input_size), bias=True)
                                    for filter_size in self.filter_sizes])

    def forward(self, word_ids, extword_ids):
        # word_ids: sentence_num * sentence_len
        # extword_ids: sentence_num * sentence_len
        # batch_masks: sentence_num * sentence_len
        sen_num, sent_len = word_ids.shape

        # word_embed: sentence_num * sentence_len * 100
        # 根据 index 取出词向量
        word_embed = self.word_embed(word_ids)
        extword_embed = self.extword_embed(extword_ids)
        batch_embed = word_embed + extword_embed

        if self.training:
            batch_embed = self.dropout(batch_embed)
        # batch_embed: sentence_num x 1 x sentence_len x 100
        # squeeze 是为了添加一个 channel 的维度，成为 B * C * H * W
        # 方便下面做 卷积
        batch_embed.unsqueeze_(1)

        pooled_outputs = []
        # 通过 3 个卷积核做 3 次卷积核池化
        for i in range(len(self.filter_sizes)):
            # 通过池化公式计算池化后的高度: o = (i-k)/s+1
            # 其中 o 表示输出的长度
            # k 表示卷积核大小
            # s 表示步长，这里为 1
            filter_height = sent_len - self.filter_sizes[i] + 1  # 卷积后尺寸的变化
            # conv：sentence_num * out_channel * filter_height * 1
            conv = self.convs[i](batch_embed)
            hidden = F.relu(conv)
            # 定义池化层
            # (filter_height, filter_width)
            mp = nn.MaxPool2d((filter_height, 1))
            # pooled：sentence_num * out_channel * 1 * 1 -> sen_num * out_channel
            # 也可以通过 squeeze 来删除无用的维度
            pooled = mp(hidden).reshape(sen_num,
                                        self.out_channel)

            pooled_outputs.append(pooled)
        # 拼接 3 个池化后的向量
        # reps: sen_num * (3*out_channel)
        reps = torch.cat(pooled_outputs, dim=1)

        if self.training:
            reps = self.dropout(reps)

        return reps


sent_hidden_size = 256
sent_num_layers = 2


# build model
class Model(nn.Module):
    def __init__(self, vocab, emb_vocab, label_encoder):
        super(Model, self).__init__()
        self.sent_rep_size = emb_vocab.word_dim * 3
        self.doc_rep_size = sent_hidden_size * 2
        self.all_parameters = {}
        parameters = []
        self.word_encoder = WordCNNEncoder(vocab, emb_vocab)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.word_encoder.parameters())))

        self.sent_encoder = SentEncoder(self.sent_rep_size)
        self.sent_attention = Attention(self.doc_rep_size)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))

        self.out = nn.Linear(self.doc_rep_size, label_encoder.label_size, bias=True)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.out.parameters())))

        if use_cuda:
            self.to(device)

        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters  # key(str): list

        logging.info('Build model with cnn word encoder, lstm sent encoder.')

        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])  # 参数总数
        logging.info('Model param num: %.2f M.' % (para_num / 1e6))

    def forward(self, batch_inputs):
        # batch_inputs(batch_inputs1, batch_inputs2): b x doc_len x sent_len
        # batch_masks : b x doc_len x sent_len
        batch_inputs1, batch_inputs2, batch_masks = batch_inputs
        batch_size, max_doc_len, max_sent_len = batch_inputs1.shape[0], batch_inputs1.shape[1], batch_inputs1.shape[2]
        batch_inputs1 = batch_inputs1.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_inputs2 = batch_inputs2.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_masks = batch_masks.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len

        sent_reps = self.word_encoder(batch_inputs1, batch_inputs2)  # sen_num x sent_rep_size

        sent_reps = sent_reps.view(batch_size, max_doc_len, self.sent_rep_size)  # b x doc_len x sent_rep_size
        batch_masks = batch_masks.view(batch_size, max_doc_len, max_sent_len)  # b x doc_len x max_sent_len
        sent_masks = batch_masks.bool().any(2).float()  # b x doc_len

        sent_hiddens = self.sent_encoder(sent_reps, sent_masks)  # b x doc_len x doc_rep_size
        doc_reps, atten_scores = self.sent_attention(sent_hiddens, sent_masks)  # b x doc_rep_size

        batch_outputs = self.out(doc_reps)  # b x num_labels

        return batch_outputs
