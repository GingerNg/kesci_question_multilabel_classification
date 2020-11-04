
# lines = ["121", "4545", "34343"]
# lines = list(map(lambda x: x.strip(), lines))
# vocab = dict(zip(lines, range(len(lines))))
# print(vocab)
from utils import file_utils
from nlp_tools import word_utils, keyword_tool
from corpus_preprocess.industry_corpus_process import Industries
words = []
for label in Industries:
    topic_word_path = "/home/wujinjie/Language-Model/data/topic_model/lda_%s.pkl"

    bunch = file_utils.read_bunch(path=topic_word_path % label)
    # print(bunch)
    words.append(bunch)
    # print(bunch.keys())

text = "网龙银行品牌诞生于1910，具有百年历史，成立以来，我们坚持在传承中提升、在创新中发展、在革命中跨越，秉承“特色发展内生增长、创新驱动”的经验方针，致力于建设“特色化、智慧化、综合化、国际化”一流上市好银行。"
keywords = keyword_tool.get_keywords(text)
print(keywords)
kws = {}
for kw in keywords:
    kws[kw[0]] = kw[1]
# segs = word_utils.segment(text).split(" ")

for ind in range(len(Industries)):
    inter_words = words[ind].keys() & kws.keys()
    print(inter_words)
    score = 0.0
    for w in inter_words:
        score += words[ind][w] * kws[w]
    print(Industries[ind], score)
