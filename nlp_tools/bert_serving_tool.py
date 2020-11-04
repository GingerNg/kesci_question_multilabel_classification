# -*- cobert_encg:utf-8 -*-
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity


class Encoding(object):
    def __init__(self):
        self.server_ip = "127.0.0.1"
        self.bert_client = BertClient(ip=self.server_ip, check_length=False)

    def encode(self, query):
        tensor = self.bert_client.encode([query])
        return tensor

    def encodes(self, queries):
        tensors = self.bert_client.encode(queries)
        return tensors

    def query_similarity(self, query_list):
        tensors = self.bert_client.encode(query_list)
        return cosine_similarity(tensors)[0][1]


ec = Encoding()


def bert_enc(sents):
    def isempty(sent):
        if not sent or sent.strip() == "":
            return "其它"
        else:
            return sent
    sents = list(map(isempty, sents))
    return ec.encodes(sents)

if __name__ == "__main__":
    text = "网龙银行品牌诞生于1910，具有百年历史，成立以来，我们坚持在传承中提升、在创新中发展、在革命中跨越，秉承“特色发展内生增长、创新驱动”的经验方针，致力于建设“特色化、智慧化、综合化、国际化”一流上市好银行。"
    res = bert_enc(sents=[text])
    print(res[0].shape)