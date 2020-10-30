# from utils import adjust_learning_rate_cosine, adjust_learning_rate_step
from utils.fold_split import ShuffleSlicer
from utils import file_utils, model_utils
import torch.optim as optim
import torch.nn as nn
import torch
import os
import argparse
import pandas as pd
import time
from corpus_preprocess.tianchi_news_process import fold_data_path, batch2tensor, get_examples, process_corpus_dl
from corpus_preprocess.industry_corpus_process import label_encoder, segment
from models.text_cnn import Model
from models.optimzers import Optimizer
from nlp_tools.vocab_builder import Vocab, EmbVocab
from corpus_preprocess.tianchi_news_process import data_iter
from evaluation_index.scores import get_score, reformat
import numpy as np
from sklearn.metrics import classification_report
from cfg import proj_path


def run(method="train", save_path=None, infer_texts=[]):
    shuffle_slicer = ShuffleSlicer()
    # start_time = time.time()

    raw_data_path = os.path.join(proj_path, "data/raw_data/tianchi_news/train_set.csv")
    texts = pd.read_csv(raw_data_path, sep='\t', encoding='UTF-8')
    train_df, dev_df = shuffle_slicer.split(texts, dev=False)

    # Test_data = {'label': [0] * len(texts), 'text': test_texts}

    clip = 5.0
    epochs = 100
    # log_interval = 50
    test_batch_size = 128
    train_batch_size = 128
    train_texts, train_labels = process_corpus_dl(train_df)
    Train_data = {'label': train_labels, 'text': train_texts}

    dev_texts, dev_labels = process_corpus_dl(dev_df)
    Dev_data = {'label': dev_labels, 'text': dev_texts}
    vocab = Vocab(Train_data)
    step = 0

    # 读取训练好的词向量文件
    word2vec_path = os.path.join(proj_path, 'data/emb/word2vec.txt')
    emb_vocab = EmbVocab(embfile=word2vec_path)

    def _eval(data):
        model.eval()  # 不启用 BatchNormalization 和 Dropout
        # data = dev_data
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_data in data_iter(data, test_batch_size, shuffle=False):
                torch.cuda.empty_cache()
                batch_inputs, batch_labels = batch2tensor(batch_data)
                batch_outputs = model(batch_inputs)
                y_pred.extend(torch.max(batch_outputs, dim=1)
                              [1].cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())

            score, dev_f1 = get_score(y_true, y_pred)
        return score, dev_f1

    def _infer(data):
        model.eval()
        # data = dev_data
        y_pred = []
        with torch.no_grad():
            for batch_data in data_iter(data, test_batch_size, shuffle=False):
                torch.cuda.empty_cache()
                batch_inputs, batch_labels = batch2tensor(batch_data)
                batch_outputs = model(batch_inputs)
                y_pred.extend(torch.max(batch_outputs, dim=1)
                              [1].cpu().numpy().tolist())
                print(label_encoder.label2name(y_pred))

    if method == "train":

        # 生成模型可处理的格式
        train_data = get_examples(Train_data, vocab, emb_vocab, label_encoder)
        dev_data = get_examples(Dev_data, vocab, emb_vocab, label_encoder)
        # 一个epoch的batch个数
        batch_num = int(np.ceil(len(train_data) / float(train_batch_size)))

        model = Model(vocab, emb_vocab, label_encoder)
        optimizer = Optimizer(model.all_parameters, None)  # 优化器

        #　loss
        criterion = nn.CrossEntropyLoss()  # obj
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
            # batch_idx = 1
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

                y_pred.extend(torch.max(batch_outputs, dim=1)
                              [1].cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())

                nn.utils.clip_grad_norm_(
                    optimizer.all_params, max_norm=clip)  # 梯度裁剪
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
            # if set(y_true) == set(y_pred):
            #     print("report")
            #     report = classification_report(y_true, y_pred, digits=4, target_names=label_encoder.target_names)
            #     # logging.info('\n' + report)
            #     print(report)

            # eval
            _, dev_f1 = _eval(data=dev_data)

            if best_dev_f1 <= dev_f1:
                best_dev_f1 = dev_f1
                early_stop = 0
                best_train_f1 = train_f1
                save_path = model_utils.save_checkpoint(
                    model, epoch, save_folder=os.path.join(proj_path, "data/textcnn/tianchi_news"))
                print("save_path:{}".format(save_path))
                # torch.save(model.state_dict(), save_model)
            else:
                early_stop += 1
                if early_stop == EarlyStopEpochs:  # 达到早停次数，则停止训练
                    break

            print("score:{}, dev_f1:{}, best_train_f1:{}, best_dev_f1:{}".format(
                dev_f1, score, best_train_f1, best_dev_f1))
    else:
        model = model_utils.load_checkpoint(save_path)
        if method == "test":
            test_data_path = 'data/raw_data/tianchi_news/test_a.csv'
            test_data_file = os.path.join(proj_path, test_data_path)
            f = pd.read_csv(test_data_file, sep='\t', encoding='UTF-8')
            texts = f['text'].tolist()
            Test_data = {'label': [0] * len(texts), 'text': texts}

            test_data = get_examples(Test_data, vocab, emb_vocab, label_encoder)
            # model.load_state_dict(torch.load(save_model))
            _, dev_f1 = _eval(data=test_data)
            print(dev_f1)

        elif method == "infer":
            infer_texts = list(map(segment, infer_texts))
            # print(infer_texts)
            Infer_data = {'label': [0] * len(infer_texts), 'text': infer_texts}
            infer_data = get_examples(Infer_data, vocab, emb_vocab, label_encoder)
            _infer(data=infer_data)


if __name__ == "__main__":
    run(method="test",
        save_path=os.path.join(proj_path, "data/textcnn/epoch_17.pth"),
        infer_texts=[])
