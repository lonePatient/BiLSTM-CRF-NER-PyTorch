#encoding:utf-8
import argparse
import pandas as pd
from pyner.utils.logginger import init_logger
from pyner.config.basic_config import configs as config
from pyner.model.embedding import word2vec

def main():
    logger = init_logger(log_name='word2vec',log_dir=config['log_dir'])
    logger.info('load %s data from disk '%args['tag'])
    train_data = pd.read_csv(config['raw_train_path'],usecols=['word'])
    test_data = pd.read_csv(config['raw_test_path'],usecols=['word'])
    data = pd.concat([train_data,test_data],axis =0)

    logger.info("initializing emnedding model")
    word2vec_model = word2vec.Word2Vec(size   = args['size'],
                                       window = args['window'],
                                       min_count=3,
                                       tag = args['tag'],
                                       save_dir = config['embedding_dir'],
                                       logger   = logger)
    logger.info('train %s word2vec embedding'%args['tag'])
    word2vec_model.train_w2v([[word for word in document.item().split()] for document in list(data.values)])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='PyTorch model training')
    ap.add_argument('-t',
                    '--tag',
                    required=True,
                    type = str,
                    help = 'Seed for initializing training.')
    ap.add_argument('-s',
                    '--size',
                    required=True,
                    default=300,
                    type = int,
                    help = 'Batch size for dataset iterators')
    ap.add_argument('-w',
                    '--window',
                    default=5,
                    type = int,
                    help = 'Batch size for dataset iterators')

    args = vars(ap.parse_args())
    main()

    '''
    python train_word2vec.py --tag=word --size=300
    python train_word2vec.py --tag=word --size=250
    python train_word2vec.py --tag=char --size=300
    python train_word2vec.py --tag=char --size=250
    
    '''