from torch import nn
import torch
import logging
import torch.nn.functional as F
# 读取训练好的词向量文件
word2vec_path = '../emb/word2vec.txt'
dropout = 0.15

# 输入是：
# 输出是：


class WordCNNEncoder(nn.Module):
    def __init__(self, vocab):
        super(WordCNNEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.word_dims = 100  # 词向量的长度是 100 维
        # padding_idx 表示当取第 0 个词时，向量全为 0
        # 这个 Embedding 层是可学习的
        self.word_embed = nn.Embedding(
            vocab.word_size, self.word_dims, padding_idx=0)

        # pretrained_embs
        extword_embed = vocab.load_pretrained_embs(word2vec_path)
        extword_size, word_dims = extword_embed.shape
        logging.info("Load extword embed: words %d, dims %d." %
                     (extword_size, word_dims))

        # # 这个 Embedding 层是不可学习的
        self.extword_embed = nn.Embedding(
            extword_size, word_dims, padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
        self.extword_embed.weight.requires_grad = False  # 不对这个变量求梯度

        input_size = self.word_dims

        self.filter_sizes = [2, 3, 4]  # n-gram window
        self.out_channel = 100
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
        batch_embed = word_embed + extword_embed  # 串联

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
            filter_height = sent_len - self.filter_sizes[i] + 1
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

class Model(nn.Module):
    pass

# build optimizer
learning_rate = 2e-4
decay = .75
decay_step = 1000


class Optimizer:
    def __init__(self, model_parameters):
        self.all_params = []
        self.optims = []
        self.schedulers = []

        for name, parameters in model_parameters.items():
            if name.startswith("basic"):
                optim = torch.optim.Adam(parameters, lr=learning_rate)
                self.optims.append(optim)

                l = lambda step: decay ** (step // decay_step)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=l)
                self.schedulers.append(scheduler)
                self.all_params.extend(parameters)

            else:
                Exception("no nameed parameters.")

        self.num = len(self.optims)

    def step(self):
        for optim, scheduler in zip(self.optims, self.schedulers):
            optim.step()
            scheduler.step()
            optim.zero_grad()

    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()

    def get_lr(self):
        lrs = tuple(map(lambda x: x.get_lr()[-1], self.schedulers))
        lr = ' %.5f' * self.num
        res = lr % lrs
        return res