import codecs
import numpy as np


def load_pretrained_vec(pretrained_path, vec_len):
    pre_trained = {}
    for i, line in enumerate(codecs.open(pretrained_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        # print(line)
        if len(line) == vec_len + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
    return pre_trained


char_pre_trained = None

# word_pre_trained = load_pretrained_vec("/opt/project/Jupyter/StaticFile/sgns.financial.word",vec_len=300)


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


def lookup(chars):
    """[summary]

    Args:
        char_pre_trained ([type]): [description]
        chars ([str]): [description]

    Returns:
        [type]: [description]
    """
    global char_pre_trained
    if char_pre_trained is None:
        char_pre_trained = load_pretrained_vec(
            "/home/ginger/Projects/StaticResource/char_wiki_100.utf8", vec_len=100)
    new_array = np.zeros((100))
    res = []
    chars = chars.replace(" ", "")
    for char in chars:
        if char in char_pre_trained:
            res.append(char_pre_trained[char])
    if len(res) >= 30:
        res = res[0:30]
    else:
        while len(res) < 30:
            res.append(new_array)
    # x_train = sequence.pad_sequences(res, maxlen=30)
    res = np.array(res)
    # return res.T
    # return len(res.sum(axis=0))
    return res.sum(axis=0)


def char_matrix2vec(matrix):
    return matrix.sum(axis=1)


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


if __name__ == "__main__":
    chars = "摩旅中国西部纪录片"
    print(lookup(chars))
