import random
import numpy as np
import pandas as pd
from keras.utils.np_utils import *
# from nlp_tools import vec_tool
from nlp_tools.word_utils import segment

Industries = [ '互联网',
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


def process_corpus_fasttext(data_df):
    """
    将数据集处理成fasttext模型需要的格式
    :param corpus_path:
    :param target_label:
    :return:
    """
    datas = []
    for line in data_df:
        print(line)
        words = segment(line["content"])
        if len(words) > 1:
            keyword = line["keyword"]
            label = str(Industries.index(keyword)+1)
            new_line = "__label__" + label + " "
            datas.append(new_line + words)
    name = ['sentence']
    data_pd = pd.DataFrame(columns=name, data=datas)
    return data_pd
