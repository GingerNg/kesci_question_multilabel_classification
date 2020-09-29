import random
import numpy as np
import pandas as pd
from keras.utils.np_utils import *
from nlp_tools import vec_tool
from nlp_tools.word_utils import segment


########################
# 数据集处理
########################

# Names = ["A", "B", "C", "D", "E", "F"]
def hebin_cols(*args):
    """
    """
    res = []
    args = list(args)
    for ind, arg in enumerate(args):
        if arg == 1:
            res.append(str(ind+1))
    # args = list(map(str, args))
    # return ",".join(args)
    return res


def hebin_list(*args):
    """
    """
    return list(args)


def hebin2str(*args):
    args = list(map(str, args))
    return "-".join(args)

# data_utils.hebin_cols


def process_label(data_df, func):
    res = data_df.apply(lambda row: func(row['category_A'],
                                         row['category_B'],
                                         row['category_C'],
                                         row['category_D'],
                                         row['category_E'],
                                         row['category_F'],), axis=1)
    return res

################
# 根据模型要求处理数据，特征编码，标签编码
################


def process_corpus_fasttext(data_df):
    """
    将数据集处理成fasttext模型需要的格式
    :param corpus_path:
    :param target_label:
    :return:
    """
    datas = []
    for line in data_df.values.tolist():
        new_line = ""
        words = segment(line[7])
        for i in range(1, 7):
            if str(line[i]) == "1":
                new_line += "__label__" + str(i) + " "
        datas.append(new_line + words)
    name = ['sentence']
    data_pd = pd.DataFrame(columns=name, data=datas)
    return data_pd


def process_corpus_nn(data_df):
    """[数据预处理，label处理]
    Args:
        data_df ([type]): [description]
        np.array(data_df["sent_vecs"].tolist())
    """
    data_df["sent_vecs"] = data_df["Question Sentence"].apply(vec_tool.lookup)
    data_df["label"] = process_label(data_df, hebin_list)
    return data_df




class CorpusProcessor(object):
    pass


class NNCorpusProcessor(object):
    pass


class FasttextCorpusProcessor(object):

    def process_corpus(self, corpus_path, labels):
        """
        构造任务数据集+分词
        :param corpus_path:
        :param target_label:
        :return:
        """
        datas = []
        corpus_data = pd.read_csv(corpus_path, encoding="utf-8", sep=",")
        corpus_data = corpus_data.dropna()
        for label in labels:
            target_data = corpus_data[corpus_data["label"]
                                      == label].sentence.values.tolist()
            for content in target_data:
                try:
                    words = segment(content)
                    datas.append("__label__" + str(label) +
                                 " " + " ".join(words))
                except Exception as e:
                    print(e)
        name = ['sentence']
        golds_pd = pd.DataFrame(columns=name, data=datas)
        new_path = os.path.join("./data", "batch") + ".csv"
        golds_pd.to_csv(new_path, encoding='utf-8')
        return new_path

    def train_test_corpus(corpus_path, sample_info=None):
        """
        有监督任务
        分训练集和测试集
        """
        corpus_data = pd.read_csv(corpus_path, encoding="utf-8", sep=",")
        sentences = corpus_data.sentence.values.tolist()
        random.shuffle(sentences)  # 做乱序处理，使得同类别的样本不至于扎堆
        all = len(sentences)
        train_len = int(0.8*all)
        # train_len = all
        print(train_len)
        print(all-train_len)
        if "train_path" in sample_info:
            writeData(sentences[0:train_len], sample_info["train_path"])
        if "test_path" in sample_info:
            writeData(sentences[train_len:all], sample_info["test_path"])
        train = sentences[0:all]
        test = sentences[0:all]

        new_sample_info = {
            "train_num": all,
            "test_num": all
        }
        if sample_info is not None:
            new_sample_info.update(sample_info)
        return train, test, new_sample_info

    # def train_test_corpus(self,corpus_path, sample_info=None):
    #     """
    #     sample_info: fold info
    #     :param corpus_path:
    #     :param sample_info:
    #     :return:
    #     """
    #     data_info = []
    #     for i in range(0,67):
    #         data_info.append(0)
    #     corpus_data = pd.read_csv(corpus_path, encoding="utf-8", sep=",").values.tolist()
    #     texts = []
    #     labels = []
    #     # random.shuffle(corpus_data)
    #     for data in corpus_data:
    #         texts.append(data[2])
    #         if data[1] == 67:
    #             data_label = data[1] - 2
    #             labels.append(data[1] - 2)
    #         else:
    #             data_label = data[1] - 1
    #             labels.append(data[1]-1)
    #         data_info[data_label] += 1
    #     # 调用to_categorical将b按照9个类别来进行转换
    #     labels = to_categorical(labels, 67)
    #     new_labels = []
    #     new_texts = []
    #     for i in range(len(labels)):
    #         if texts[i] in new_texts:
    #             index = new_texts.index(texts[i])
    #             new_labels[index] = new_labels[index] + labels[i]
    #         else:
    #             new_texts.append(texts[i])
    #             new_labels.append(labels[i])

    #     """统计每个标签对应的数据个数"""
    #     data_label_count65 = []
    #     for i in range(0,len(labels65)):
    #         data_label_count65.append([labels65[i],data_info[i]])

        # name = ['label', 'count']
        # test = pd.DataFrame(columns=name, data=data_label_count65)
        # # print(test)
        # test.to_csv("label_counts"+".csv", encoding='utf-8')

        # train_corpus = new_texts[0:len(new_texts)-1000]
        # train_label = np.array(new_labels[0:len(new_texts)-1000])
        # test_corpus = new_texts[len(new_texts)-1000:]
        # test_label = np.array(new_labels[len(new_texts)-1000:])
        # train_corpus = new_texts[sample_info[0]:sample_info[1]] + new_texts[sample_info[2]:sample_info[3]]
        # train_label = np.array(new_labels[sample_info[0]:sample_info[1]] + new_labels[sample_info[2]:sample_info[3]])
        # test_corpus = new_texts[sample_info[1]:sample_info[2]]
        # test_label = np.array(new_labels[sample_info[1]:sample_info[2]])
        # sample_info = {
        #     "neg": {"train_num": 20, "test_num": 20}
        # }
        # return train_corpus, train_label, test_corpus, test_label, sample_info
