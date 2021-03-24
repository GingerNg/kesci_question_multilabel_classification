# pytorch_text_classification



@20200913
eda 分fold --> 选择模型1(model+optimizer+Loss) --> **根据模型要求处理数据，特征编码，标签编码(corpus_preprocess)** -->  交叉验证 --> 调参
               选择模型2...

先分fold，再进行数据处理

@20201018
train：
epoch(所有数据) -> batch/slice（随机选择固定size的数据）
[Datawhale零基础入门NLP赛事 - Task5 基于深度学习的文本分类2-2TextCNN](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.24.64063dadx0bgpr&postId=118258)
[【详细注释+流程讲解】基于深度学习的文本分类 TextCNN](https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.27.6406224eBph8DC&postId=122736)
[Datawhale零基础入门NLP赛事 - Task6 基于深度学习的文本分类3-BERT](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.75.6406224ek2VNsN&postId=118260)



##### DatasetProcessor
- get_examples  文本和标签 处理 组成成obj
- data_iter    批， 迭代器
- batch2tensor   按batch生成tensor


id--label--name


# 模型放置在gpu上进行计算
if torch.cuda.is_available():
    model.cuda()

doc -- sentence -- word

@todo
结合知识图谱的文本分类

### 语言模型的使用
##### huggingface
https://github.com/huggingface/transformers

from transformers import BertModel

Bertpooler 其实就是将BERT的[CLS]的hidden_state 取出，经过一层DNN和Tanh计算后输出。
https://zhuanlan.zhihu.com/p/120315111


##### bert as service
https://github.com/hanxiao/bert-as-service
<!-- # pip3 install bert-serving-server  # server
# pip3 install bert-serving-client
# 启动server端：bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4
# note: tensorflow 版本不能太高 tensorflow==1.14.0 -->
