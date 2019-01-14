#encoding:utf-8
import os
import multiprocessing
from gensim.models import word2vec

class Word2Vec():
    def __init__(self,size,
                 save_dir,
                 logger,
                 window = 5,
                 min_count = 2,
                 sg=0,
                 hs=0,
                 iter = 10,
                 tag = 'word',
                 seed = 2018):
        '''
        :param size: 词向量的纬度，一般在100-200
        :param window: 窗口大小
        :param min_count: 对词进行过滤，默认是5
        :param sg: sg=1表示skip-gram，对低频词敏感；sg=0表示CBOW算法
        :param hs: hs=1表示使用层级softmax
        :param workers:
        :return:
        '''
        self.size=size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.hs = hs
        self.workers = multiprocessing.cpu_count()
        self.save_dir = save_dir
        self.tag=tag
        self.logger = logger
        self.seed = seed
        self.iter = iter

    def train_w2v(self, data):
        """
        训练wv模型
        :param filename:path
        :return:none
        """
        # sentences = word2vec.LineSentence(data)  #[wordlist,wordlist......]
        self.logger.info('train word2vec....')
        self.logger.info('word vector size is: %d'%self.size)
        model = word2vec.Word2Vec(data,
                                  size=self.size,
                                  window=self.window,
                                  hs=self.hs,
                                  sg=self.sg,
                                  min_count=self.min_count,
                                  workers=self.workers,
                                  seed=self.seed,
                                  iter= self.iter)
        model_file = '%s_word2vec_size%d_win%d'%(self.tag,self.size,self.window) +'.txt'
        self.logger.info('saveing word2vec model ....')

        # model.save(os.path.join(self.save_dir,model_file))
        # 以文本形式保存
        with open(os.path.join(self.save_dir,model_file),'w') as fw:
            for word in model.wv.vocab:
                vector = model[word]
                fw.write(str(word) + ' ' + ' '.join(map(str, vector)) + '\n')
