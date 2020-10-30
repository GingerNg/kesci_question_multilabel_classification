import codecs
import numpy as np
from .keyword_tool import get_keywords
from cfg import word_emb_path, char_emb_path, baike_vec_path
import h5py

char_pre_trained = None
word_pre_trained = None
baike_pre_trained = None


def load_hifile(pretrained_path):
    # hfile = h5py.File('w2v.h5')
    hfile = h5py.File(pretrained_path)
    id2vec = hfile['data'][:]  # id2vec
    hfile.close()
    return id2vec


def load_pretrained_vec(pretrained_path, vec_len):
    """
    """
    pre_trained = {}
    for i, line in enumerate(codecs.open(pretrained_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        # print(line)
        if len(line) == vec_len + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
    return pre_trained


def resize_weight(keywords):
    """
    归一化权重
    """
#     print(type(keywords))
    a = 0.0
    new_ks = []
    for k in keywords:
        #         print(k)
        a += k[1]
    for k in keywords:
        new_ks.append((k[0], k[1]/a))
    return new_ks


def lookup(chars, train_type="char"):
    """[summary]

    Args:
        char_pre_trained ([type]): [description]
        chars ([str]): [description]

    Returns:
        [type]: [description]
    """
    global char_pre_trained
    global word_pre_trained
    global baike_pre_trained
    if train_type == "char":
        if char_pre_trained is None:
            char_pre_trained = load_pretrained_vec(char_emb_path, vec_len=100)
        # new_array = np.zeros((100))
        res = []
        chars = chars.replace(" ", "")
        for char in chars:
            if char in char_pre_trained:
                res.append(char_pre_trained[char])
        if len(res) >= 30:
            res = res[0:30]
        # else:
        #     while len(res) < 30:
        #         res.append(new_array)
    elif train_type == "baike":
        if baike_pre_trained is None:
            baike_pre_trained = load_hifile(baike_vec_path)
        # new_array = np.zeros((50))
        res = []
        # chars = chars.replace(" ", "")
        # words = chars.split(" ")
        word_ids = chars
        for word_id in word_ids:
            # if word in word_pre_trained:
            res.append(baike_pre_trained[word_id])
        if len(res) >= 30:
            res = res[0:30]
    else:
        if word_pre_trained is None:
            word_pre_trained = load_pretrained_vec(word_emb_path, vec_len=50)
        # new_array = np.zeros((50))
        res = []
        # chars = chars.replace(" ", "")
        words = chars.split(" ")
        for word in words:
            if word in word_pre_trained:
                res.append(word_pre_trained[word])
        if len(res) >= 30:
            res = res[0:30]
        # else:
        #     while len(res) < 30:
        #         res.append(new_array)
    # x_train = sequence.pad_sequences(res, maxlen=30)
    size = len(res)
    res = np.array(res)
    # return res.T
    # return len(res.sum(axis=0))
    return res.sum(axis=0)/size


def char_matrix2vec(matrix):
    return matrix.sum(axis=1)


def weight_word_embed(keywords):
    """
    短语向量表示：加权词向量
    """
    global baike_pre_trained
    vecs = []
#     print(keywords)
    keywords = resize_weight(keywords)
    if baike_pre_trained is None:
        baike_pre_trained = load_hifile(baike_vec_path)
#     print(keywords)
    for keyword in keywords:
        vecs.append(baike_pre_trained[keyword[0]]*keyword[1])
    return np.array(vecs).sum(axis=0)


def weight_char_embed(chars, top_k=5, vec_len=100):
    """
    短语向量表示：加权词向量
    """
    vec = np.zeros((vec_len))
    chars = chars.replace(" ", "")
    keywords = get_keywords(chars)
    keywords = keywords[0:top_k]
#     print(keywords)
    keywords = resize_weight(keywords)
#     print(keywords)
    for keyword in keywords:
        for char in keyword[0]:
            if char in char_pre_trained.keys():
                vec += char_pre_trained[char]*keyword[1]
    return vec


def batch_wce(lines):
    return list(map(weight_char_embed, lines))


def get_inter_vocabs(pretrained_vocabs, text_vocab):
    """
    词向量字典和文本字典的交集
    """
    vocab = pretrained_vocabs & text_vocab
    return vocab


if __name__ == "__main__":
    chars = "摩旅中国西部纪录片"
    print(lookup(chars))
