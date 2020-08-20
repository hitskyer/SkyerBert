import time
import torch
import numpy as np
from importlib import import_module
import argparse
import utils
import train

parser = argparse.ArgumentParser(description='Skyer-Bert-Text-Classsification')
parser.add_argument('--model', type=str, default='SkyerBertDPCNN',
                    help = 'choose a model SkyerBert, SkyerBertCNN, SkyerBertRNN, SkyerBertRCNN, SkyerBertDPCNN')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews' #数据集地址
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True #保证每次运行结果一样

    start_time = time.time()
    print('加载数据集')
    train_data, dev_data, test_data = utils.build_dataset(config)
    train_iter = utils.build_iterator(train_data, config)
    dev_iter = utils.build_iterator(dev_data, config)
    test_iter = utils.build_iterator(test_data, config)

    time_diff = utils.get_time_diff(start_time)
    print("模型开始之前，准备数据时间：", time_diff)

    # 模型训练，评估与测试
    model = x.Model(config).to(config.device)
    train.train(config, model, train_iter, dev_iter, test_iter)
    #train.test(config, model, test_iter)
