#encoding:utf-8
import os
import random
import operator
import numpy as np
from tqdm import tqdm
from collections import Counter
from ..utils.utils import pkl_read,pkl_write,text_write

'''
1. 建立语料库
2. 将句子中的词转化为对应id
3. 初始化embedding matrix
'''
class DataTransformer(object):
    '''
    数据类型:句子，类别

    sent,class
    'I am a student','o o o o'

    中间使用','进行分割
    '''
    def __init__(self,
                 vocab_path,
                 logger,
                 max_features = None,
                 min_freq = 3,
                 all_data_path = None,
                 label_to_id = None,
                 train_file = None,
                 valid_file = None,
                 test_file = None,
                 valid_size = None,
                 skip_header = False,
                 is_train_mode = True,
                 seed=1024,
                 ):

        self.seed = seed
        self.logger = logger
        self.valid_size = valid_size
        self.min_freq = min_freq
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.all_data_path = all_data_path
        self.vocab_path = vocab_path
        self.skip_header = skip_header
        self.max_features = max_features
        self.label_to_id = label_to_id
        self.is_train_mode = is_train_mode

    def _split_sent(self,line):
        """
        句子处理成单词
        :param line: 原始行
        :return: 单词， 标签
        """
        res = line.strip('\n').split()
        return res

    def _word_to_id(self,word, vocab):
        """
        单词-->ID
        :param word: 单词
        :param word2id: word2id @type: dict
        :return:
        """
        return vocab[word] if word in vocab else vocab['<unk>']

    def train_val_split(self,X, y, valid_size=0.3, random_state=2018, shuffle=True):

        self.logger.info('train val split')
        data = []
        for data_x, data_y in tqdm(zip(X, y), desc='Merge'):
            data.append((data_x, data_y))
        del X, y
        N = len(data)
        test_size = int(N * valid_size)
        if shuffle:
            random.seed(random_state)
        random.shuffle(data)
        valid = data[:test_size]
        train = data[test_size:]
        return train,valid

    def build_vocab(self):
        if os.path.isfile(self.vocab_path):
            self.vocab = pkl_read(self.vocab_path)
        else:
            count = Counter()
            with open(self.all_data_path, 'r') as fr:
                self.logger.info('Building word vocab')
                for i,line in enumerate(fr):
                    # 数据首行为列名
                    if i==0 and self.skip_header:
                        continue
                    words = self._split_sent(line)
                    count.update(words)
            count = {k: v for k, v in count.items()}
            count = sorted(count.items(), key=operator.itemgetter(1))
            # 词典
            all_words = [w[0] for w in count if w[1] >= self.min_freq]
            if self.max_features:
                all_words = all_words[:self.max_features]
            # 一些特殊词
            flag_words = ['<pad>', '<unk>']
            all_words = flag_words + all_words
            self.logger.info('vocab_size is %d' % len(all_words))
            # 词典到编号的映射
            word2id = {k: v for k, v in zip(all_words, range(0, len(all_words)))}
            assert word2id['<pad>'] == 0, "ValueError: '<pad>' id is not 0"
            # 写入文件中
            pkl_write(data = word2id,filename=self.vocab_path)
            self.vocab = word2id

    def sentence2id(self,raw_data_path=None,raw_target_path  =None,x_var = None,y_var = None):
        """
        将word转化为对应的id
        :param valid_size: 验证集大小
        """
        self.logger.info('sentence to id')
        if self.is_train_mode:
            if os.path.isfile(self.train_file) and os.path.isfile(self.valid_file):
                return True
            sentences, labels = [], []
            with open(raw_data_path, 'r') as fr_x,open(raw_target_path,'r') as fr_y:
                for i,(sent,target) in enumerate(zip(fr_x,fr_y)):
                    if i==0 and self.skip_header:
                        continue
                    words = self._split_sent(sent)
                    label = self._split_sent(target)
                    if len(words) ==0 or len(label) ==0:
                        continue
                    sent2id = [self._word_to_id(word=word, vocab=self.vocab) for word in words]
                    label = [self.label_to_id[x] for x in label]
                    sentences.append(sent2id)
                    labels.append(label)
            # 分割数据集
            if self.valid_size:
                train, val = self.train_val_split(X = sentences, y = labels,
                                                  valid_size=self.valid_size,
                                                  random_state=self.seed,
                                                  shuffle=True)
                text_write(self.train_file, train,x_var = x_var,y_var = y_var)
                text_write(self.valid_file, val,x_var = x_var,y_var = y_var)
        else:
            if os.path.isfile(self.test_file):
                return True
            sentences,labels = [],[]
            with open(raw_data_path, 'r') as fr_x:
                for i,sent in enumerate(fr_x):
                    if i==0 and self.skip_header:
                        continue
                    words = self._split_sent(sent)
                    if len(words) ==0:
                        continue
                    sent2id = [self._word_to_id(word=word, vocab=self.vocab) for word in words]
                    label    = [-1 for _ in range(len(sent2id))]
                    sentences.append(sent2id)
                    labels.append(label)
            text_write(self.test_file,zip(sentences,labels),x_var = x_var,y_var = y_var)

    def build_embedding_matrix(self,embedding_path,emb_mean = None,emb_std = None):
        '''
        构建词向量权重矩阵
        :param embedding_path:
        :param embedding_dim:
        :param oov_type:
        :return:
        '''
        self.logger.info("initializer embedding matrix")
        embeddings_index = self._load_embedding(embedding_path)
        all_embs = np.stack((embeddings_index.values()))
        if emb_mean is None or emb_std is None:
            emb_mean = all_embs.mean()
            emb_std  = all_embs.std()
        embed_size = all_embs.shape[1]
        nb_words = len(self.vocab)
        # 这里我们简单使用正态分布产生随机值
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, id in self.vocab.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[id] = embedding_vector

        return embedding_matrix

    def _load_embedding(self, embedding_path):
        '''
        加载pretrained
        :param embedding_path:
        :return:
        '''
        self.logger.info(" load emebedding weights")
        embeddings_index = {}
        f = open(embedding_path, 'r',errors='ignore',encoding = 'utf8')
        for line in f:
            values = line.split(' ')
            try:
                word  = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except:
                print("Error on ", values[:2])
        f.close()
        self.logger.info('Total %s word vectors.' % len(embeddings_index))
        return embeddings_index
