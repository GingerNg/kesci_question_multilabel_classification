import os, sys
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(current_path)
sys.path.append(current_path)
os.chdir("..")
import pandas as pd
import logging
import numpy as np
import pickle
from utils import file_utils
from utils.model_utils import use_cuda, device
import torch
from nlp_tools.doc_utils import sentence_split
from collections import Counter


class LabelEncoer():  # 标签编码
    def __init__(self):
        self.unk = 1
        self._id2label = []
        self.target_names = []
        # process label
        label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政',
                      5: '社会', 6: '教育', 7: '财经', 8: '家居', 9: '游戏',
                      10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}

        for label, name in label2name.items():
            self._id2label.append(label)
            self.target_names.append(name)
        # self.label_counter = Counter(data['label'])
        # for label in range(len(self.label_counter)):
            # count = self.label_counter[label]
            # self._id2label.append(label)
            # self.target_names.append(label2name[label])

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)


label_encoder = LabelEncoer()


# 语料相关配置
fold_data_path = os.path.join(current_path, "data/textcnn/data/fold_data.pl")


def process_corpus_fasttext(data_df):
    """
    将数据集处理成fasttext模型需要的格式
    :param corpus_path:
    :param target_label:
    :return:
    """
    datas = []

    def _process(line):
        # print(line)
        words = line["text"]
        if len(words) > 1:
            label = str(line["label"])
            new_line = "__label__" + label + " "
            datas.append(new_line + words)
    if isinstance(data_df, list):
        for line in data_df:
            _process(line)
    else:
        for ind, row in data_df.iterrows():
            # print(type(row))
            _process(row.to_dict())
    name = ['sentence']
    data_pd = pd.DataFrame(columns=name, data=datas)
    return data_pd


# split data to 10 fold
fold_num = 10
data_file = './data/raw_data/tianchi_news/train_set.csv'


def all_data2fold(fold_num, num=10000):
    fold_data = []
    f = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
    texts = f['text'].tolist()[:num]
    labels = f['label'].tolist()[:num]

    total = len(labels)   # 数据总量

    indexs = list(range(total))  # 　下标-索引
    np.random.shuffle(indexs)

    all_texts = []  # 文本
    all_labels = []
    for i in indexs:
        all_texts.append(texts[i])
        all_labels.append(labels[i])

    label2id = {}  # 　{label1:[], label2:[]}
    for i in range(total):  # 下标-索引
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = [i]
        else:
            label2id[label].append(i)

    all_index = [[] for _ in range(fold_num)]
    for label, data in label2id.items():
        # print(label, len(data))
        batch_size = int(len(data) / fold_num)  # size_per_fold
        other = len(data) - batch_size * fold_num  # 余数
        for i in range(fold_num):
            cur_batch_size = batch_size + 1 if i < other else batch_size
            # print(cur_batch_size)
            batch_data = [data[i * batch_size + b]
                          for b in range(cur_batch_size)]
            all_index[i].extend(batch_data)

    batch_size = int(total / fold_num)
    other_texts = []
    other_labels = []
    other_num = 0
    start = 0
    for fold in range(fold_num):
        num = len(all_index[fold])
        texts = [all_texts[i] for i in all_index[fold]]
        labels = [all_labels[i] for i in all_index[fold]]

        if num > batch_size:
            fold_texts = texts[:batch_size]
            other_texts.extend(texts[batch_size:])
            fold_labels = labels[:batch_size]
            other_labels.extend(labels[batch_size:])
            other_num += num - batch_size
        elif num < batch_size:
            end = start + batch_size - num
            fold_texts = texts + other_texts[start: end]
            fold_labels = labels + other_labels[start: end]
            start = end
        else:
            fold_texts = texts
            fold_labels = labels

        assert batch_size == len(fold_labels)

        # shuffle
        index = list(range(batch_size))
        np.random.shuffle(index)

        shuffle_fold_texts = []
        shuffle_fold_labels = []
        for i in index:
            shuffle_fold_texts.append(fold_texts[i])
            shuffle_fold_labels.append(fold_labels[i])

        # {'label': [], 'text': []}
        data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}
        fold_data.append(data)

    logging.info("Fold lens %s", str([len(data['label']) for data in fold_data]))
    file_utils.writeBunch(path=fold_data_path, bunchFile=fold_data)
    return fold_data


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))  # ceil 向上取整
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]  ## ???

        yield docs   #　返回一个batch的数据


def data_iter(data, batch_size, shuffle=True, noise=1.0):
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

        lengths = [example[1] for example in data]
        noisy_lengths = [- (l + np.random.uniform(- noise, noise)) for l in lengths]
        sorted_indices = np.argsort(noisy_lengths).tolist()
        sorted_data = [data[i] for i in sorted_indices]
    else:
        sorted_data = data

    batched_data.extend(list(batch_slice(sorted_data, batch_size))) # [[],[]]

    if shuffle:
        np.random.shuffle(batched_data)

    for batch in batched_data:
        yield batch



def get_examples(data, vocab, emb_vocab, label_encoder, max_sent_len=256, max_segment=8):
    """[summary]
        dict --> list
        word--> id
    Args:
        data ([type]): [    label:[], text:[[],...,[]]   ]
        vocab ([type]): [description]
        max_sent_len (int, optional): [description]. Defaults to 256.
        max_segment (int, optional): [description]. Defaults to 8.

    Returns:
        [list]: [label_id, ]
    """

    label2id = label_encoder.label2id
    examples = []

    for text, label in zip(data['text'], data['label']):
        # label
        id = label2id(label)

        # words
        sents_words = sentence_split(text, vocab, max_sent_len, max_segment)
        doc = []
        for sent_len, sent_words in sents_words:
            word_ids = vocab.word2id(sent_words)
            extword_ids = emb_vocab.extword2id(sent_words)
            doc.append([sent_len, word_ids, extword_ids])  # sent_len 句子长度：即句子中词个数
        examples.append([id, len(doc), doc])   # len(doc): 文档中句子的个数

    logging.info('Total %d docs.' % len(examples))
    return examples


def batch2tensor(batch_data):
    '''
        [[label, doc_len, [[sent_len, [sent_id0, ...], [sent_id1, ...]], ...]]
    '''
    batch_size = len(batch_data)
    doc_labels = []
    doc_lens = []
    doc_max_sent_len = []
    for doc_data in batch_data:
        doc_labels.append(doc_data[0])
        doc_lens.append(doc_data[1])
        sent_lens = [sent_data[0] for sent_data in doc_data[2]]
        max_sent_len = max(sent_lens)  #　最大句子长度
        doc_max_sent_len.append(max_sent_len)

    # max_doc_len = max(doc_lens)  # 最大文档长度（句子个数）
    # max_sent_len = max(doc_max_sent_len)
    # print(max_doc_len, max_sent_len)
    max_sent_len = 256
    max_doc_len = 8

    # 初始化矩阵: batch_inputs1 batch_inputs2 双输入
    batch_inputs1 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
    batch_inputs2 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
    batch_masks = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.float32)
    batch_labels = torch.LongTensor(doc_labels)

    for b in range(batch_size):
        for sent_idx in range(doc_lens[b]):
            sent_data = batch_data[b][2][sent_idx]
            for word_idx in range(sent_data[0]):   #　sent_data[0]: 句子长度
                batch_inputs1[b, sent_idx, word_idx] = sent_data[1][word_idx]
                batch_inputs2[b, sent_idx, word_idx] = sent_data[2][word_idx]
                batch_masks[b, sent_idx, word_idx] = 1

    if use_cuda:
        batch_inputs1 = batch_inputs1.to(device)
        batch_inputs2 = batch_inputs2.to(device)
        batch_masks = batch_masks.to(device)
        batch_labels = batch_labels.to(device)

    return (batch_inputs1, batch_inputs2, batch_masks), batch_labels


if __name__ == "__main__":
    fold_data = all_data2fold(10)
