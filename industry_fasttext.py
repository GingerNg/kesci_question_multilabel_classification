from sklearn import utils
from fasttext import train_supervised
# import fasttext
from nlp_tools.word_utils import segment
from corpus_preprocess.industry_corpus_process import process_corpus_fasttext
from models.ft import print_results
import pymongo
import pandas as pd
from utils import file_utils, data_utils


# client = pymongo.MongoClient('192.168.235.223', 27017)
# db = client['nlp_db']
# baidu_text = db['baidu_text']

raw_data_path = "data/raw_data/baidu/nlp_db.baidu_text.csv"
# texts = data_utils.df2list(pd.read_csv(raw_data_path))
texts = pd.read_csv(raw_data_path)
# texts = list(text_pd)
# print(texts[0])
#texts = list(baidu_text.find())

data_pd = process_corpus_fasttext(texts)

data_pd.to_csv("./data/fasttext/industry_ft.csv")


utils.shuffle(data_pd)
train_len = data_pd.shape[0]//5 * 4
train_df = data_pd[0:train_len]
test_df = data_pd[train_len:]
train_path = "./data/fasttext/industry_train_df_ft.txt"
test_path = "./data/fasttext/industry_test_df_ft.txt"
file_utils.write_data(train_df["sentence"].to_list(), train_path)
file_utils.write_data(test_df["sentence"].to_list(), test_path)
# train_df.to_csv(train_path)
# test_df.to_csv(test_path)

classifier = train_supervised(
    input=train_path, epoch=500, lr=0.01, wordNgrams=2, verbose=2, minCount=1,
    # loss="hs"
)

print_results(*classifier.test(train_path))

print_results(*classifier.test(test_path))
