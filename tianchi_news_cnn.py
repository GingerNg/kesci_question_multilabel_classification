# from utils import adjust_learning_rate_cosine, adjust_learning_rate_step
import torch.optim as optim
import torch.nn as nn
import torch
import os
import argparse
import pandas as pd
from utils import file_utils
from corpus_preprocess.tianchi_news_process import fold_data_path, batch2tensor, current_path, get_examples
import time


start_time = time.time()

fold_data = file_utils.readBunch(path=fold_data_path)
# build train, dev, test data
fold_id = 9

# dev
Dev_data = fold_data[fold_id]

# train
train_texts = []
train_labels = []
for i in range(0, fold_id):
    data = fold_data[i]
    train_texts.extend(data['text'])
    train_labels.extend(data['label'])

Train_data = {'label': train_labels, 'text': train_texts}

# test
test_data_path = 'data/raw_data/tianchi_news/test_a.csv'
test_data_file = os.path.join(current_path, test_data_path)
f = pd.read_csv(test_data_file, sep='\t', encoding='UTF-8')
texts = f['text'].tolist()
Test_data = {'label': [0] * len(texts), 'text': texts}


clip = 5.0
epochs = 1
early_stops = 3
log_interval = 50

test_batch_size = 128
train_batch_size = 128

save_model = './cnn.bin'
save_test = './cnn.csv'

from models.text_cnn import Model
from models.optimzers import Optimizer
from nlp_tools.vocab_builder import Vocab, EmbVocab
from corpus_preprocess.tianchi_news_process import data_iter
from evaluation_index.scores import get_score, reformat
import numpy as np
from sklearn.metrics import classification_report

step = 0

# 读取训练好的词向量文件
word2vec_path = os.path.join(current_path, 'data/emb/word2vec.txt')
emb_vocab = EmbVocab(embfile=word2vec_path)
vocab = Vocab(Train_data)
model = Model(vocab)
optimizer = Optimizer(model.all_parameters, None)  # 优化器

#　loss
criterion = nn.CrossEntropyLoss()  # obj

# 生成模型可处理的格式
train_data = get_examples(Train_data, vocab)
test_data = get_examples(Test_data, vocab)
dev_data = get_examples(Dev_data, vocab)

batch_num = int(np.ceil(len(train_data) / float(train_batch_size)))  # 一个epoch的batch个数
if __name__ == "__main__":
    best_train_f1, best_dev_f1 = 0, 0
    early_stop = -1
    EarlyStopEpochs = 3  # 当多个epoch，dev的指标都没有提升，则早停
    # train
    print("start train")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        model.train()  # 启用 BatchNormalization 和 Dropout
        overall_losses = 0
        losses = 0
        batch_idx = 1
        y_pred = []
        y_true = []
        for batch_data in data_iter(train_data, train_batch_size, shuffle=True):
            torch.cuda.empty_cache()
            batch_inputs, batch_labels = batch2tensor(batch_data)
            batch_outputs = model(batch_inputs)
            loss = criterion(batch_outputs, batch_labels)
            loss.backward()

            loss_value = loss.detach().cpu().item()
            losses += loss_value
            overall_losses += loss_value

            y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
            y_true.extend(batch_labels.cpu().numpy().tolist())

            nn.utils.clip_grad_norm_(optimizer.all_params, max_norm=clip)  # 梯度裁剪
            for cur_optim, scheduler in zip(optimizer.optims, optimizer.schedulers):
                cur_optim.step()
                scheduler.step()
            optimizer.zero_grad()
            step += 1
            # print(step)
        print(epoch)
        overall_losses /= batch_num
        overall_losses = reformat(overall_losses, 4)
        score, train_f1 = get_score(y_true, y_pred)
        print("score:{}, train_f1:{}".format(train_f1, score))
        if set(y_true) == set(y_pred):
            print("report")
            report = classification_report(y_true, y_pred, digits=4, target_names=vocab.target_names)
            # logging.info('\n' + report)
            print(report)

        # eval
        model.eval()  # 不启用 BatchNormalization 和 Dropout
        data = dev_data
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_data in data_iter(data, test_batch_size, shuffle=False):
                torch.cuda.empty_cache()
                batch_inputs, batch_labels = batch2tensor(batch_data)
                batch_outputs = model(batch_inputs)
                y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())

            score, dev_f1 = get_score(y_true, y_pred)

        if best_dev_f1 <= dev_f1:
            best_dev_f1 = dev_f1
            early_stop = 0
            best_train_f1 = train_f1
        else:
            early_stop += 1
            if early_stop == EarlyStopEpochs:  # 达到早停次数，则停止训练
                break

        print("score:{}, dev_f1:{}".format(dev_f1, score))
            # during_time = time.time() - start_time

            # if test:
            #     df = pd.DataFrame({'label': y_pred})
            #     df.to_csv(save_test, index=False, sep=',')
            # else:
            #     logging.info(
            #         '| epoch {:3d} | dev | score {} | f1 {} | time {:.2f}'.format(epoch, score, f1,
            #                                                                   during_time))
            #     if set(y_true) == set(y_pred) and self.report:
            #         report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
            #         logging.info('\n' + report)


