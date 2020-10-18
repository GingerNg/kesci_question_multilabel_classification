from utils import adjust_learning_rate_cosine, adjust_learning_rate_step
import torch.optim as optim
import torch.nn as nn
import torch
import os
import argparse
import pandas as pd
from utils import file_utils
from corpus_preprocess.tianchi_news_process import fold_data_path

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

train_data = {'label': train_labels, 'text': train_texts}

# test
test_data_file = '../data/test_a.csv'
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

vocab = Vocab(train_data)
model = Model(vocab)
optimizer = Optimizer(model.all_parameters)
if __name__ == "__main__":
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
            batch_inputs, batch_labels = self.batch2tensor(batch_data)
            batch_outputs = model(batch_inputs)
            loss = criterion(batch_outputs, batch_labels)
            loss.backward()

            loss_value = loss.detach().cpu().item()
            losses += loss_value
            overall_losses += loss_value

            y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
            y_true.extend(batch_labels.cpu().numpy().tolist())

            nn.utils.clip_grad_norm_(self.optimizer.all_params, max_norm=clip)
            for optimizer, scheduler in zip(self.optimizer.optims, self.optimizer.schedulers):
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad()





            step += 1