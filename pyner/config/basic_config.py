#encoding:utf-8
from os import path
import multiprocessing

BASE_DIR = 'pyner'

configs = {
    'all_data_path': path.sep.join([BASE_DIR,'dataset/raw/source_BIO_2014_cropus.txt']),   # 总的数据，一般是将train和test何在一起构建语料库
    'raw_train_path': path.sep.join([BASE_DIR,'dataset/raw/source_BIO_2014_cropus.txt']),  # 原始的训练数据
    'raw_target_path': path.sep.join([BASE_DIR,'dataset/raw/target_BIO_2014_cropus.txt']), #　原始的标签数据
    'raw_test_path': path.sep.join([BASE_DIR,'dataset/raw/test.txt']),           # 原始的test数据集

    'train_file_path': path.sep.join([BASE_DIR,'dataset/processed/train.json']), # 处理之后的训练数据
    'valid_file_path': path.sep.join([BASE_DIR,'dataset/processed/valid.json']),   #　valid数据
    'test_file_path': path.sep.join([BASE_DIR,'dataset/processed/test.json']),   # test数据
    'embedding_weight_path': path.sep.join([BASE_DIR, # 词向量
                                            'output/embedding/min_count_1_window_5_300d.word2vec']),
    'glove_weight_path': path.sep.join([BASE_DIR, # 词向量
                                            'output/embedding/glove_vectors_word.txt']),
    'vocab_path': path.sep.join([BASE_DIR,'dataset/processed/vocab.pkl']), # 语料数据
    'result_path': path.sep.join([BASE_DIR, 'output/result/predict_result.txt']),# test预测结果

    'log_dir': path.sep.join([BASE_DIR, 'output/log']), # 模型运行日志
    'writer_dir': path.sep.join([BASE_DIR, 'output/TSboard']),# TSboard信息保存路径
    'figure_dir': path.sep.join([BASE_DIR, 'output/figure']),# 图形保存路径
    'checkpoint_dir': path.sep.join([BASE_DIR, 'output/checkpoints/{arch}']),# 模型保存路径
    'embedding_dir': path.sep.join([BASE_DIR, 'output/embedding']),# 词向量保存路径

    'valid_size': 0.1, # valid数据集大小
    'min_freq': 1,     # 最小词频，构建语料
    'num_classes': 12, # 类别个数 这里主要还有pad 0
    'max_length': 80,  # word文本平均长度,按照覆盖95%样本的标准，取截断长度:np.percentile(list,95.0)
    'max_features': 100000, # how many unique words to use (i.e num rows in embedding vector)
    'embedding_dim':300,   # how big is each word vector

    'batch_size': 256,   # how many samples to process at once
    'epochs': 100,       # number of epochs to train
    'start_epoch': 1,
    'learning_rate': 0.015,
    'weight_decay': 5e-4, # 权重衰减因子，防止过拟合
    'n_gpus': [0], # GPU个数,如果只写一个数字，则表示gpu标号从0开始，并且默认使用gpu:0作为controller,
                   # 如果以列表形式表示，即[1,3,5],则我们默认list[0]作为controller
    'x_var':'source', # 原始文本字段名
    'y_var':'target', # 原始标签字段名
    'num_workers': multiprocessing.cpu_count(), # 线程个数
    'seed': 2018,     # seed
    'lr_patience': 5, # number of epochs with no improvement after which learning rate will be reduced.
    'mode': 'min',    # one of {min, max}
    'monitor': 'val_loss',  # 计算指标
    'early_patience': 10,   # early_stopping
    'save_best_only': True, # 是否保存最好模型
    'best_model_name': '{arch}-best2.pth', #保存文件
    'epoch_model_name': '{arch}-{epoch}-{val_loss}.pth', #以epoch频率保存模型
    'save_checkpoint_freq': 10, #保存模型频率，当save_best_only为False时候，指定才有作用
    'label_to_id': {    # 标签映射
        "B_PER": 1,  # 人名
        "I_PER": 2,
        "B_LOC": 3,  # 地点
        "I_LOC": 4,
        "B_ORG": 5,  # 机构
        "I_ORG": 6,
        "B_T": 7,    # 时间
        "I_T": 8,
        "O": 9,      # 其他
        "BOS": 10,   # 起始符
        "EOS": 11    # 结束符
    },
    # 模型列表以及模型配置信息
    'models': {'bilstm_crf':{'hidden_size': 200,
                             'bi_tag': True,
                             'dropout_p':0.5,
                             'dropout_emb':0.0,
                             'num_layer': 1,
                             'use_cuda':True}
              }
}
