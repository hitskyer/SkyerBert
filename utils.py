from tqdm import tqdm
import torch
import time
from datetime import timedelta
import pickle as pkl
import os
import sys

PAD, CLS = '[PAD]', '[CLS]'

def load_dataset(file_path, config):
    """
    返回结果 4个list ids, label, ids_len, mask
    :param file_path:
    :param seq_len:
    :return:
    """
    contents = []
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            try:
                content, label = line.strip().split('\t')
            except:
                continue
            token = config.tokenizer.tokenize(content)
            token = [CLS] +token
            
            pad_size = config.pad_size
            token_ids = config.tokenizer.convert_tokens_to_ids(token)
            mask = []
            if pad_size > 0:
                if len(token) < pad_size:
                    mask = [1]*len(token_ids) + [0]*(pad_size-len(token))
                    token_ids = token_ids + [0]*(pad_size-len(token))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
            else:
                sys.stderr.write("utils::load_dataset(..) failed.\n")
                sys.exit(-1)
            seq_len = min(len(token), pad_size)
            contents.append((token_ids, int(label), seq_len, mask))
    return contents

def build_dataset(config):
    """
    返回值 train, dev ,test
    :param config:
    :return:
    """
    if os.path.exists(config.datasetpkl):
        dataset = pkl.load(open(config.datasetpkl, 'rb'))
        train = dataset['train']
        dev = dataset['dev']
        test = dataset['test']
    else:
        train = load_dataset(config.train_path, config)
        dev = load_dataset(config.dev_path, config)
        test = load_dataset(config.test_path, config)
        dataset = {}
        dataset['train'] = train
        dataset['dev'] = dev
        dataset['test'] = test
        pkl.dump(dataset, open(config.datasetpkl, 'wb'))
    return train, dev, test

class DatasetIterator(object):
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset

        self.batch_size = batch_size
        self.n_batches = len(dataset) // batch_size
        self.residue = False #记录batch数量是否为整数
        if len(dataset) % self.n_batches != 0:
            self.residue = True

        self.index = 0
        self.device = device
    def _to_tensor(self, datas):
        x = torch.LongTensor([item[0] for item in datas]).to(self.device) #样本数据ids
        y = torch.LongTensor([item[1] for item in datas]).to(self.device) #标签数据label

        seq_len = torch.LongTensor([item[2] for item in datas]).to(self.device) #每一个序列的真实长度
        mask = torch.LongTensor([item[3] for item in datas]).to(self.device)

        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size : len(self.dataset)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[self.index * self.batch_size : (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterator(dataset, config.batch_size, config.device)
    return iter

def get_time_diff(start_time):
    """
    获取已经使用的时间
    :param start_time:
    :return:
    """
    end_time = time.time()
    time_diff = end_time - start_time
    return timedelta(seconds=int(round(time_diff)))
