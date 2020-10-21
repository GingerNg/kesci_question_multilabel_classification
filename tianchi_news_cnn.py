# from utils import adjust_learning_rate_cosine, adjust_learning_rate_step
import torch.optim as optim
import torch.nn as nn
import torch
import os
import argparse
import pandas as pd
from utils import file_utils
from corpus_preprocess.tianchi_news_process import fold_data_path, batch2tensor, current_path, get_examples

fold_data = file_utils.readBunch(path=fold_data_path)
# build train, dev, test data
fold_id = 9

# dev
dev_data = fold_data[fold_id]

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
test_data = {'label': [0] * len(texts), 'text': texts}


clip = 5.0
epochs = 1
early_stops = 3
log_interval = 50

test_batch_size = 128
train_batch_size = 128

save_model = './cnn.bin'
save_test = './cnn.csv'

from models.text_cnn import Optimizer, Model
from models.vocab_builder import Vocab
from corpus_preprocess.tianchi_news_process import data_iter
from evaluation_index.scores import get_score, reformat
import numpy as np
from sklearn.metrics import classification_report

step = 0
vocab = Vocab(Train_data)
model = Model(vocab)
optimizer = Optimizer(model.all_parameters)
criterion = nn.CrossEntropyLoss()
train_data = get_examples(Train_data, vocab)
batch_num = int(np.ceil(len(train_data) / float(train_batch_size)))
if __name__ == "__main__":
    # train
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        model.train()
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

            nn.utils.clip_grad_norm_(optimizer.all_params, max_norm=clip)
            for cur_optim, scheduler in zip(optimizer.optims, optimizer.schedulers):
                cur_optim.step()
                scheduler.step()
            optimizer.zero_grad()
            step += 1
            # print(step)
        print(epoch)
        overall_losses /= batch_num
        overall_losses = reformat(overall_losses, 4)
        score, f1 = get_score(y_true, y_pred)
        if set(y_true) == set(y_pred):
            report = classification_report(y_true, y_pred, digits=4, target_names=vocab.target_names)
            # logging.info('\n' + report)

