import random
import numpy as np
import pandas as pd
from collections import Counter
# from keras.utils.np_utils import *
# from nlp_tools import vec_tool
from nlp_tools.word_utils import segment
from utils.model_utils import use_cuda, device

# labels
Industries = ['互联网',
              '交通运输',
              '体育竞技',
              '党政机关',
              '医疗保健',
              '婚礼婚庆',
              '宠物行业',
              '房产建筑',
              # '房产行业',
              '教育行业',
              '旅游行业',
              '美容保养',
              '节日庆典',
              '运动健身',
              '金融投资',
              '餐饮行业']


class LabelEncoer():  # 标签编码
    def __init__(self, labels):
        super().__init__()
        self.unk = 1
        self._id2label = range(len(labels))
        self.target_names = labels
        def reverse(x): return dict(zip(x, range(len(x))))  # 词与id的映射
        self._label2id = reverse(self._id2label)
        # self._id2name =

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)

    def name2label(self, x):
        return self.target_names.index(x)

    def label2name(self, xs):
        if isinstance(xs, list):
            return [self.target_names[x] for x in xs]
        return self.target_names[xs]

    @property
    def label_size(self):
        return len(self._id2label)

        # process label
        # label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政',
        #               5: '社会', 6: '教育', 7: '财经', 8: '家居', 9: '游戏',
        #               10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}

        # self.label_counter = Counter(data['label'])

        # for label in range(len(self.label_counter)):
        #     count = self.label_counter[label]
        #     self._id2label.append(label)
        #     self.target_names.append(label2name[label])


label_encoder = LabelEncoer(Industries)

def process_corpus_fasttext(data_df):
    """
    将数据集处理成fasttext模型需要的格式
    :param corpus_path:
    :param target_label:
    :return:
    """
    datas = []

    def _process(line):
        print(line)
        words = segment(line["content"])
        if len(words) > 1:
            keyword = line["keyword"]
            label = str(Industries.index(keyword)+1)
            new_line = "__label__" + label + " "
            datas.append(new_line + words)
    if isinstance(data_df, list):
        for line in data_df:
            _process(line)
    else:
        for ind, row in data_df.iterrows():
            # print(type(row))
            _process(row.to_dict())
        # print(line)
        # words = segment(line["content"])
        # if len(words) > 1:
        #     keyword = line["keyword"]
        #     label = str(Industries.index(keyword)+1)
        #     new_line = "__label__" + label + " "
        #     datas.append(new_line + words)
    name = ['sentence']
    data_pd = pd.DataFrame(columns=name, data=datas)
    return data_pd


def process_corpus_dl(data_df, seg="true"):
    texts = []
    labels = []

    def _process(line):
        # print(line)
        if isinstance(line["content"], str):
            if seg:
                words = segment(line["content"])
            else:
                words = line["content"]
            # print(words)
            if len(words) > 1:
                keyword = line["keyword"]
                labels.append(label_encoder.name2label(keyword))
                texts.append(words)
    if isinstance(data_df, list):
        for line in data_df:
            _process(line)
    else:
        for _, row in data_df.iterrows():
            # print(type(row))
            _process(row.to_dict())

    return texts, labels


from nlp_tools import bert_serving_tool


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))  # ceil 向上取整
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - \
            1 else len(data) - batch_size * i
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]  # ???

        yield docs  # 　返回一个batch的数据

def data_iter_bert_nn(data, batch_size, shuffle=True, noise=1.0):
    """[    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch]
        input:
        [sent_len, word_ids, extword_ids]
    Yields:
        [type]: [description]
    """
    batched_data = []
    if shuffle:
        np.random.shuffle(data)
        sorted_data = data
        # lengths = [example[1] for example in data]
        # noisy_lengths = [- (l + np.random.uniform(- noise, noise)) for l in lengths]
        # sorted_indices = np.argsort(noisy_lengths).tolist()
        # sorted_data = [data[i] for i in sorted_indices]
    else:
        sorted_data = data

    batch = list(batch_slice(sorted_data, batch_size))
    batched_data.extend(batch)  # [[],[]]

    if shuffle:
        np.random.shuffle(batched_data)

    for batch in batched_data:
        yield batch


def get_example_bert_nn(data, label_encoder):
    label2id = label_encoder.label2id
    examples = []
    for text, label in zip(data['text'], data['label']):
        # label
        id = label2id(label)

        doc = bert_serving_tool.bert_enc(text)[0]
        examples.append([id, len(doc), doc])

    # logging.info('Total %d docs.' % len(examples))
    return examples


import torch
def batch2tensor(batch_data):
    batch_size = len(batch_data)
    doc_labels = []
    for doc_data in batch_data:
        doc_labels.append(doc_data[0])

    # batch_inputs = torch.zeros((batch_size, 768), dtype=torch.int64)
    batch_labels = torch.LongTensor(doc_labels)
    inputs = []
    for b in range(batch_size):
        inputs.append(batch_data[b][2])
        # a = torch.from_numpy(batch_data[b][2])
        # batch_inputs[b, ] = a
    batch_inputs = torch.from_numpy(np.array(inputs))
    # print(batch_inputs.shape)

    if use_cuda:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

    return (batch_inputs), batch_labels
