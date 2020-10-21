import jieba
from torch.utils.data import Dataset, DataLoader
import torch
from pytorch_transformers import OpenAIGPTTokenizer, OpenAIGPTConfig


special_tokens = ['<bos>', '<del>', '<eos>', '<pad>']
tokenizer = OpenAIGPTTokenizer.from_pretrained(
    'openai-gpt', special_tokens=special_tokens)


def convert_tokens_to_ids(text):
    return tokenizer.convert_tokens_to_ids(text)


class MyData(Dataset):

    def __init__(self, texts, labels, is_train=True):
        self.texts = [jieba.lcut(t) for t in texts]
        self.labels = labels
        # 其他操作 ....

    def __getitem__(self, item):
        token_id = convert_tokens_to_ids(self.texts[item])  # 词 -> token_id
        label = self.labels[item]
        return torch.LongTensor(token_id), torch.LongTensor([label])

    def __len__(self):
        return len(self.texts)


def get_dataloader(dataset, batch_size, shuffle=False, drop_last=False):
    data_iter = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return data_iter


# dataset = MyData(texts, labels)
# dataloader = get_dataloader(dataset, batch_size=16)  # 成功封装成迭代器
