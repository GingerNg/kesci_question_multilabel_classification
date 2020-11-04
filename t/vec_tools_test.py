import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)
os.chdir("..")
import jieba
from cfg import baike_vec_path, baike_vocab_path
from nlp_tools import vec_tool, word_utils, vocab_builder
from utils import distances

# jieba.load_userdict(baike_vocab_path)
# festivals = ['互联网', '交通运输', '体育竞技', '党政机关', '医疗保健', '婚礼婚庆', '宠物行业', '房产建筑',
#        '教育行业', '旅游行业', '美容保养', '节日庆典', '运动健身', '金融投资', '餐饮行业']
festivals = ['元旦', '春节', '清明节', '劳动节', '端午节',
             '中秋节', '国庆节', '七夕节', '教师节', '植树节', '圣诞节', '万圣节']


if __name__ == "__main__":
    vecs = []
    # baike_vec = vec_tool.load_hifile(baike_vec_path)
    baike_vocab = vocab_builder.BaikeVocab(vocab_path=baike_vocab_path)
    for fes in festivals:
        print(fes, " ".join(jieba.cut(fes)))
        word_ids = baike_vocab.word2id(" ".join(jieba.cut(fes)).split())
        # print(word_ids)
        vecs.append(vec_tool.lookup(word_ids, train_type="baike"))

    # for fes in festivals:
    #     print(fes, word_utils.segment(fes))
    #     vecs.append(vec_tool.lookup(word_utils.segment(fes), train_type="word"))
    #     # print(vec_tool.lookup(fes))
    dists = []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            print(festivals[i], festivals[j], distances.euclidean_dist(vecs[i], vecs[j]))
            dists.append(distances.euclidean_dist(vecs[i], vecs[j]))
    print(min(dists), max(dists))
    # vec1 = vec_tool.lookup('圣诞节')
    # vec2 = vec_tool.lookup('圣诞')
