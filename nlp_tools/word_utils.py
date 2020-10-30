import jieba
from cfg import stop_word_path
from utils import file_utils
import re

stopWordList = None


def remain(string):
    # string = "123我123456abcdefgABCVDFF？/ ，。,.:;:''';'''[]{}()（）《》"
    # print(string)
    sub_str = re.sub(
        u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a|。])", "", string)
    # print(sub_str)
    return sub_str


def remove_stopword(word):
    global stopWordList
    word = remain(word)
    if stopWordList is None:
        stopWordList = get_stop_word(stop_word_path)
    if word in stopWordList:
        return False
    else:
        return True


def segment(mytext):
    """中文分词"""
    if isinstance(mytext, str) and len(mytext) > 0:
        return " ".join(filter(remove_stopword, jieba.cut(mytext)))
    else:
        return ""


def get_stop_word(inputFile):
    stopWordList = file_utils.read_file(inputFile).splitlines()
    return stopWordList
