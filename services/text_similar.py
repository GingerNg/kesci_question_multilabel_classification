import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)
os.chdir("..")
from nlp_tools.keyword_tool import get_keywords
from nlp_tools import vocab_builder, vec_tool, tfidf_tool
from cfg import baike_vocab_path
from utils import distances
import jieba

baike_vocab = vocab_builder.BaikeVocab(vocab_path=baike_vocab_path)
festivals = ['元旦', '春节', '清明节', '劳动节', '端午节',
             '中秋节', '国庆节', '七夕节', '教师节', '植树节', '圣诞节', '万圣节']
vecs = []
for fes in festivals:
    # print(fes, " ".join(jieba.cut(fes)))
    word_ids = baike_vocab.word2id(" ".join(jieba.cut(fes)).split())
    # print(word_ids)
    vecs.append(vec_tool.lookup(word_ids, train_type="baike"))


def similar():
    raw_path = os.path.join(current_path, "data/raw_data/ppt/corpus.txt")
    lines = open(raw_path, "r").readlines()
    for line in lines:
        # keywords = tfidf_tool.cal_tf(list(jieba.cut(line)))
        keywords = get_keywords(line)
        print(keywords)
        keywords = [(baike_vocab.word2id(x[0]), x[1]) for x in keywords]
        # print(keywords)
        text_vec = vec_tool.weight_word_embed(keywords[0:7])
        # print(text_vec)
        # print(keywords)
        # word_ids = baike_vocab.word2id(keywords)
        # print(word_ids)
        # text_vec = vec_tool.lookup(word_ids, train_type="baike")
        dists = []
        for i in range(len(vecs)):
            # print(festivals[i], distances.euclidean_dist(vecs[i], text_vec))
            dists.append((festivals[i], distances.euclidean_dist(vecs[i], text_vec)))
        # print(min(dists), max(dists))
        # print(dists)
        print(sorted(dists, key=lambda x: x[1])[0])

if __name__ == "__main__":
    similar()