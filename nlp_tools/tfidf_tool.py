from collections import Counter
import math
from typing import List


def cal_tf(text: List):
    """
    """
    word_dict = dict(Counter(text))
    # print(word_dict)
    return [(k, v) for k, v in word_dict.items()]


def cal_idf(ts: List):
    """
    idf = log(文档总数/包含该词的文档总数)

    Args:
        ts (List): [多个文档的词list: [[..,.,..,],....[]]
    """
    words = []
    for t in ts:
        words += list(set(t))
    word_dict = dict(Counter(words))
    # ts = [list(set(t)) for t in ts]
    idf = {}
    for k, v in word_dict.items():
        idf[k] = math.log(len(ts)/(v+1))
    return idf


def top_percent(kv, percent=0.8):
    kvs = sorted(kv.items(), key=lambda x: x[1], reverse=True)
    return kvs[0: int(len(kvs)*percent)]


if __name__ == "__main__":
    t1 = ["中国", "蜜蜂", "养殖", "中国"]
    t2 = ["美国", "牛", "养殖"]
    t2 = ["美国", "牛", "养殖"]
    ts = [t1, t2]
    print(cal_idf(ts))
    cal_tf(t1)
