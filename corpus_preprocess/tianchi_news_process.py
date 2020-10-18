import pandas as pd
import logging
import os
import numpy as np
import pickle
from utils import file_utils
os.chdir("..")

# 语料相关配置
fold_data_path = "./data/textcnn/data/fold_data.pl"


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

    total = len(labels)

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
    batch_num = int(np.ceil(len(data) / float(batch_size))) # ceil 向上取整
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]  ## ???

        yield docs


def data_iter(data, batch_size, shuffle=True, noise=1.0):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
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


# build dataset
def sentence_split(text, vocab, max_sent_len=256, max_segment=16):
    words = text.strip().split()
    document_len = len(words)

    index = list(range(0, document_len, max_sent_len))
    index.append(document_len)

    segments = []
    for i in range(len(index) - 1):
        segment = words[index[i]: index[i + 1]]
        assert len(segment) > 0
        segment = [word if word in vocab._id2word else '<UNK>' for word in segment]
        segments.append([len(segment), segment])

    assert len(segments) > 0
    # 截断
    if len(segments) > max_segment:
        segment_ = int(max_segment / 2)
        return segments[:segment_] + segments[-segment_:]  # 前后各截1/2
    else:
        return segments


def get_examples(data, vocab, max_sent_len=256, max_segment=8):
    # dict --> list
    label2id = vocab.label2id
    examples = []

    for text, label in zip(data['text'], data['label']):
        # label
        id = label2id(label)

        # words
        sents_words = sentence_split(text, vocab, max_sent_len, max_segment)
        doc = []
        for sent_len, sent_words in sents_words:
            word_ids = vocab.word2id(sent_words)
            extword_ids = vocab.extword2id(sent_words)
            doc.append([sent_len, word_ids, extword_ids])
        examples.append([id, len(doc), doc])

    logging.info('Total %d docs.' % len(examples))
    return examples


if __name__ == "__main__":
    fold_data = all_data2fold(10)
