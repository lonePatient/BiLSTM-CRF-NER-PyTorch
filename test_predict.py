#encoding:utf-8
import os
import argparse
import torch
import warnings
from pyner.test.predicter import Predicter
from pyner.io.data_loader import DataLoader
from pyner.io.data_transformer import DataTransformer
from pyner.utils.logginger import init_logger
from pyner.utils.utils import seed_everything
from pyner.test.predict_utils import test_write
from pyner.config.basic_config import configs as config
from pyner.model.nn.bilstm_crf import Model

warnings.filterwarnings("ignore")

# 主函数
def main(arch):
    logger = init_logger(log_name=arch, log_dir=config['log_dir'])
    logger.info("seed is %d"%args['seed'])
    seed_everything(seed = args['seed'])
    checkpoint_path = os.path.join(config['checkpoint_dir'].format(arch =arch),
                                    config['best_model_name'].format(arch = arch))
    device = 'cuda:%d' % config['n_gpus'][0] if len(config['n_gpus']) else 'cpu'

    # 加载数据集
    logger.info('starting load test data from disk')
    data_transformer = DataTransformer(
                     vocab_path    = config['vocab_path'],
                     test_file     = config['test_file_path'],
                     logger        = logger,
                     skip_header   = False,
                     is_train_mode = False,
                     seed          = args['seed'])
    data_transformer.build_vocab()
    data_transformer.sentence2id(raw_data_path = config['raw_test_path'],
                                 x_var=config['x_var'],
                                 y_var=config['y_var']
                                 )
    embedding_weight = data_transformer.build_embedding_matrix(embedding_path = config['embedding_weight_path'])
    test_loader = DataLoader(logger=logger,
                        is_train_mode=False,
                        x_var = config['x_var'],
                        y_var = config['y_var'],
                        skip_header = False,
                        data_path   = config['test_file_path'],
                        batch_size  = args['batch_size'],
                        max_sentence_length = config['max_length'],
                        device = device)
    test_iter = test_loader.make_iter()
    # 初始化模型和优化器
    logger.info("initializing model")
    bilstm = Model(num_classes      = config['num_classes'],
                   embedding_dim    = config['embedding_dim'],
                   model_config     = config['models'][arch],
                   embedding_weight = embedding_weight,
                   vocab_size       = len(data_transformer.vocab),
                   device           = device)
    # 初始化模型训练器
    logger.info('predicting model....')
    predicter = Predicter(model           = bilstm,
                          logger          = logger,
                          n_gpu           = config['n_gpus'],
                          test_data       = test_iter,
                          checkpoint_path = checkpoint_path,
                          label_to_id     = config['label_to_id'])
    # 拟合模型
    predictions = predicter.predict()
    test_write(data = predictions,filename = config['result_path'],raw_text_path=config['raw_test_path'])
    # 释放显存
    if len(config['n_gpus']) > 0:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='PyTorch model predicting')
    ap.add_argument('-s',
                    '--seed',
                    default=2018,
                    type = int,
                    help = 'Seed for initializing.')
    ap.add_argument('-b',
                    '--batch_size',
                    type = int,
                    default=2,
                    help = 'Batch size for dataset iterators')
    args = vars(ap.parse_args())
    print('predict total of {} models'.format(len(config['models'])))
    for i, model_name in enumerate(config['models'].keys()):
        print('{}/{}: predict {} '.format(i + 1, len(config['models']), model_name))
        main(arch = model_name)
