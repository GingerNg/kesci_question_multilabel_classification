import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(current_path)
sys.path.append(current_path)
os.chdir("..")
import torch
from nlp_tools.tokenizers import WhitespaceTokenizer
from transformers import BertModel
from cfg import bert_path
bert = BertModel.from_pretrained(bert_path)

tokenizer = WhitespaceTokenizer(bert_path)
# sent_words = ["34", "1519", "4893", "43"]
sent_words = ["窝", "个", "三", "我"]
token_ids = tokenizer.tokenize(sent_words)
sent_len = len(token_ids)
token_type_ids = [0] * sent_len

batch_size = 16
max_doc_len = 8
max_sent_len = 64
batch_inputs1 = torch.zeros(
    (batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
batch_inputs2 = torch.zeros(
    (batch_size, max_doc_len, max_sent_len), dtype=torch.int64)

for word_idx in range(len(token_ids)):
    batch_inputs1[0, 0, word_idx] = token_ids[word_idx]
    batch_inputs2[0, 0, word_idx] = token_type_ids[word_idx]

batch_inputs1 = batch_inputs1.view(
    batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
batch_inputs2 = batch_inputs2.view(
    batch_size * max_doc_len, max_sent_len)

print(batch_inputs1.shape, token_ids)    # torch.Size([512, 256])
# input_ids = torch.tensor(token_ids)
# token_type_ids = torch.tensor(token_type_ids)
sequence_output, pooled_output = bert(input_ids=batch_inputs1, token_type_ids=batch_inputs2)
print(sequence_output.shape)  # torch.Size([512, 256, 256])
print(pooled_output.shape)   # torch.Size([512, 256])