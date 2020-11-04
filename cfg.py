import os
from enum import Enum


proj_path = os.path.dirname(__file__)

# bert_path = os.path.join(proj_path, 'data/emb/bert-mini/')
bert_path = os.path.join(proj_path, "data/emb/chinese_L-12_H-768_A-12/")
# 语料相关配置
fold_data_path = os.path.join(proj_path, "data/textcnn/data/fold_data.pl")

# emb

char_emb_path = os.path.join(proj_path, "data/wiki_100.utf8")
word_emb_path = os.path.join(proj_path, "data/emb/industry_vec.txt")

baike_vocab_path = os.path.join(proj_path, "data/BaikeWordList.txt")
baike_vec_path = os.path.join(proj_path, "data/w2v.h5")

stop_word_path = os.path.join(proj_path, "data/stopwords.txt")


class STEPTYPE(Enum):
    train = "01"
    test = "02"
    build_vocab = "03"
    infer = "04"


# print(proj_path)
methods = {
    "textcnn": "cnn_lstm_attention",
    "fasttext": "",
    "tfidf_classifier": ""
}
