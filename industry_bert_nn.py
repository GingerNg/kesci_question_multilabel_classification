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
from corpus_preprocess.tianchi_news_process import fold_data_path
from corpus_preprocess.industry_corpus_process import process_corpus_dl, label_encoder, get_example_bert_nn, batch2tensor, data_iter_bert_nn
from models.text_cnn import Model, NNModel
from models.optimzers import Optimizer
from nlp_tools.vocab_builder import Vocab, EmbVocab
from evaluation_index.scores import get_score, reformat
import numpy as np
from sklearn.metrics import classification_report
from cfg import proj_path
from utils import file_utils

"""
bert_serving + NN
"""

def run(mth="train", save_path=None, infer_texts=[]):
    shuffle_slicer = ShuffleSlicer()
    # start_time = time.time()

    raw_data_path = os.path.join(proj_path, "data/raw_data/baidu/nlp_db.baidu_text.csv")
    # texts = data_utils.df2list(pd.read_csv(raw_data_path))
    texts = pd.read_csv(raw_data_path)
    train_df, dev_df, test_df = shuffle_slicer.split(texts, dev=True)

    # Test_data = {'label': [0] * len(texts), 'text': test_texts}

    clip = 5.0
    epochs = 100
    # log_interval = 50
    test_batch_size = 128
    train_batch_size = 128
    step = 0


    def _eval(data):
        model.eval()  # 不启用 BatchNormalization 和 Dropout
        # data = dev_data
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_data in data_iter_bert_nn(data, test_batch_size, shuffle=False):
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
    if mth == "preprocess":
        train_texts, train_labels = process_corpus_dl(train_df, seg=False)
        Train_data = {'label': train_labels, 'text': train_texts}

        dev_texts, dev_labels = process_corpus_dl(dev_df, seg=False)
        Dev_data = {'label': dev_labels, 'text': dev_texts}
        # # 生成模型可处理的格式
        train_data = get_example_bert_nn(Train_data, label_encoder)
        dev_data = get_example_bert_nn(Dev_data, label_encoder)

        test_texts, test_labels = process_corpus_dl(test_df)
        Test_data = {'label': test_labels, 'text': test_texts}
        test_data = get_example_bert_nn(Test_data, label_encoder)

        processed_data = {
            "train": train_data,
            "dev": dev_data,
            "test": test_data
        }
        file_utils.write_bunch(os.path.join(proj_path, "data/bert_nn/industry/indusrty.pkl"), processed_data)

    elif mth == "train":
        processed_data = file_utils.read_bunch(os.path.join(proj_path, "data/bert_nn/industry/indusrty.pkl"))
        train_data = processed_data["train"]
        dev_data = processed_data["dev"]
        test_data = processed_data["test"]

        # 一个epoch的batch个数
        batch_num = int(np.ceil(len(train_data) / float(train_batch_size)))

        print("train_num:{}, test_num:{}, dev_num:{}, batch_num:{}".format(len(train_data), len(test_data), len(dev_data), batch_num))

        model = NNModel(label_encoder)
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
            for batch_data in data_iter_bert_nn(train_data, train_batch_size, shuffle=True):
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
                    model, epoch, save_folder=os.path.join(proj_path, "data/bert_nn/industry"))
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
        if mth == "test":

            # model.load_state_dict(torch.load(save_model))
            _, dev_f1 = _eval(data=test_data)
            print(dev_f1)

        elif mth == "infer":
            infer_texts = list(map(segment, infer_texts))
            # print(infer_texts)
            Infer_data = {'label': [0] * len(infer_texts), 'text': infer_texts}
            infer_data = get_example_bert_nn(Infer_data, label_encoder)
            _infer(data=infer_data)


if __name__ == "__main__":
    run(mth="train",
        save_path="/home/wujinjie/kesci_question_multilabel_classification/data/textcnn/epoch_35.pth",
        infer_texts=["医学，是通过科学或技术的手段处理生命的各种疾病或病变的一种学科，促进病患恢复健康的一种专业。它是生物学的应用学科，分基础医学、临床医学。从生理解剖、分子遗传、生化物理等层面来处理人体疾病的高级科学。它是一个从预防到治疗疾病的系统学科，研究领域大方向包括法医学，动物医学，中医学，口腔医学，临床医学等。",
                    "西湖，位于浙江省杭州市西湖区龙井路1号，杭州市区西部，景区总面积49平方千米，汇水面积为21.22平方千米，湖面面积为6.38平方千米。",
                    "教育又是一种思维的传授"])
