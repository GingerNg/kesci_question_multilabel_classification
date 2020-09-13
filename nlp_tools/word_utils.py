import os
import sys
cur_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
print(cur_path)
sys.path.append(cur_path)
import jieba # 导入分词库
# from . import config
from utils.file_utils import readFile
def segment(mytext):
    """中文分词"""
    return " ".join(jieba.cut(mytext))
    # return jieba.cut(mytext)

chinese_word_cut = segment

def f(*args):
    print(args)
    return args


def getStopWord(inputFile):
    stopWordList = readFile(inputFile).splitlines()
    return stopWordList

# stopWordList = getStopWord(config.stopword_path)  # 获取停用词

if __name__ == "__main__":
    # res = f(1,2,3,4)
    # for r in list(res):
    #     print(r)
    print(segment("病情描述：病人是典型的“三高”，想吃拜阿司匹林做为预防用药，但是出现过敏症状。曾经治疗情况和"))