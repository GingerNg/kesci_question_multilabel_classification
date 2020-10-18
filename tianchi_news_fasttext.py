from sklearn import utils
from fasttext import train_supervised
# import fasttext
from corpus_preprocess.tianchi_news_process import process_corpus_fasttext
from models.ft import print_results
from utils import file_utils
import pandas as pd
data_df = pd.read_csv('data/raw_data/tianchi_news/train_set.csv', sep='\t')

train_df = process_corpus_fasttext(data_df)

# data_pd.to_csv("./data/fasttext/tianchi_df_ft.csv")

# utils.shuffle(data_pd)
# train_len = data_pd.shape[0]//5 * 4
# train_df = data_pd[0:train_len]
# test_df = data_pd[train_len:]
train_path = "./data/fasttext/tianchi_train_df_ft.csv"
# test_path = "./data/fasttext/kesci_test_df_ft.csv"
file_utils.writeData(train_df["sentence"].to_list(), train_path)
# file_utils.writeData(test_df["sentence"].to_list(), test_path)

# train_df.to_csv(train_path)
# test_df.to_csv(test_path)

# train_test_corpus(new_corpus_path,sample_info={"train_path":config.train_data,
#                                                 "test_path":config.test_data})

# classifier=fasttext.supervised(saveDataFile,'classifier.model',lable_prefix='__lable__')
classifier = train_supervised(
    input=train_path, epoch=500, lr=0.01, wordNgrams=2, verbose=2, minCount=1,
    # loss="hs"
)
print_results(*classifier.test(train_path))
