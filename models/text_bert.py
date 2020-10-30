from nlp_tools.tokenizers import WhitespaceTokenizer
from torch import nn
import logging
from transformers import BertModel
from models.attentions import Attention
from models.encoders import SentEncoder
from utils.model_utils import use_cuda, device
import numpy as np
from cfg import bert_path

# build word encoder
dropout = 0.15


class WordBertEncoder(nn.Module):
    def __init__(self):
        super(WordBertEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.tokenizer = WhitespaceTokenizer(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)

        self.pooled = False
        logging.info('Build Bert encoder with pooled {}.'.format(self.pooled))

    def encode(self, tokens):
        tokens = self.tokenizer.tokenize(tokens)
        return tokens

    def get_bert_parameters(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        return optimizer_parameters

    def forward(self, input_ids, token_type_ids):
        # input_ids: sen_num x bert_len
        # token_type_ids: sen_num  x bert_len

        # sen_num x bert_len x 256, sen_num x 256
        sequence_output, pooled_output = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids)

        if self.pooled:
            reps = pooled_output
        else:
            reps = sequence_output[:, 0, :]  # sen_num x 256

        if self.training:
            reps = self.dropout(reps)

        return reps


sent_hidden_size = 256

# build model


class Model(nn.Module):
    def __init__(self, vocab, label_encoder):
        super(Model, self).__init__()
        self.sent_rep_size = 256
        self.doc_rep_size = sent_hidden_size * 2
        self.all_parameters = {}
        parameters = []
        self.word_encoder = WordBertEncoder()
        bert_parameters = self.word_encoder.get_bert_parameters()

        self.sent_encoder = SentEncoder(self.sent_rep_size)
        self.sent_attention = Attention(self.doc_rep_size)
        parameters.extend(
            list(filter(lambda p: p.requires_grad, self.sent_encoder.parameters())))
        parameters.extend(
            list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))

        self.out = nn.Linear(self.doc_rep_size, label_encoder.label_size, bias=True)
        parameters.extend(
            list(filter(lambda p: p.requires_grad, self.out.parameters())))

        if use_cuda:
            self.to(device)

        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters
        self.all_parameters["bert_parameters"] = bert_parameters

        logging.info('Build model with bert word encoder, lstm sent encoder.')

        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        logging.info('Model param num: %.2f M.' % (para_num / 1e6))

    def forward(self, batch_inputs):
        # batch_inputs(batch_inputs1, batch_inputs2): b x doc_len x sent_len
        # batch_masks : b x doc_len x sent_len
        batch_inputs1, batch_inputs2, batch_masks = batch_inputs
        batch_size, max_doc_len, max_sent_len = batch_inputs1.shape[
            0], batch_inputs1.shape[1], batch_inputs1.shape[2]
        batch_inputs1 = batch_inputs1.view(
            batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_inputs2 = batch_inputs2.view(
            batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_masks = batch_masks.view(
            batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len

        sent_reps = self.word_encoder(
            batch_inputs1, batch_inputs2)  # sen_num x sent_rep_size

        # b x doc_len x sent_rep_size
        sent_reps = sent_reps.view(batch_size, max_doc_len, self.sent_rep_size)
        # b x doc_len x max_sent_len
        batch_masks = batch_masks.view(batch_size, max_doc_len, max_sent_len)
        sent_masks = batch_masks.bool().any(2).float()  # b x doc_len

        sent_hiddens = self.sent_encoder(
            sent_reps, sent_masks)  # b x doc_len x doc_rep_size
        doc_reps, atten_scores = self.sent_attention(
            sent_hiddens, sent_masks)  # b x doc_rep_size

        batch_outputs = self.out(doc_reps)  # b x num_labels

        return batch_outputs
