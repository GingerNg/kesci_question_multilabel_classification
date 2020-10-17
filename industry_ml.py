from sklearn import metrics
from sklearn.pipeline import make_pipeline  # 导入make_pipeline方法
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import utils
import pandas as pd
from nlp_tools.word_utils import segment
from models import ml_models


# client = pymongo.MongoClient('192.168.235.223', 27017)
# db = client['nlp_db']
# baidu_text = db['baidu_text']
# texts = list(baidu_text.find())
raw_data_path = "data/raw_data/baidu/nlp_db.baidu_text.csv"
texts = pd.read_csv(raw_data_path)
contents = []
titles = []
keywords = []
sentences = []
counts = []
# for text in texts:
for _, text in texts.iterrows():
    if isinstance(text["title"], str) and isinstance(text["content"], str):
        sentences.append(text["title"]+text["content"])
        # titles.append(text["title"])
        # contents.append(text["content"])
        keywords.append(text["keyword"])
        counts.append(1)
df = pd.DataFrame({"Label": keywords,
                   "Sentence": sentences,
                   "Count": counts})

df = utils.shuffle(df)

train_len = df.shape[0]//4 * 3
train_df = df[0:train_len]
test_df = df[train_len:]

train_df["CutSentence"] = train_df["Sentence"].apply(segment)
test_df["CutSentence"] = test_df["Sentence"].apply(segment)
# train_df["CutSentence"].head()


# vect = CountVectorizer()  # 实例化
vect = TfidfVectorizer()
vect.fit_transform(train_df["CutSentence"])

# nb = MultinomialNB()
# model = ml_models.bayes()
model = ml_models.svm()

pipe = make_pipeline(vect, model)

pipe.fit(train_df["CutSentence"], train_df["Label"])

y_pred = pipe.predict(test_df["CutSentence"])
print(metrics.accuracy_score(test_df["Label"], y_pred))
print(metrics.f1_score(test_df["Label"], y_pred, average="micro"))
print(metrics.f1_score(test_df["Label"], y_pred, average="macro"))
