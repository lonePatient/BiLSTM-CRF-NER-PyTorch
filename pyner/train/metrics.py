#encoding:utf-8
import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report
from ..test.predict_utils import get_entity
from collections import Counter

class Accuracy(object):
    '''
    计算准确度
    可以使用topK参数设定计算K准确度
    '''
    def __init__(self,topK):
        super(Accuracy,self).__init__()
        self.topK = topK

    def __call__(self, output, target):
        batch_size = target.size(0)
        _, pred = output.topk(self.topK, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.topK].view(-1).float().sum(0)
        result = correct_k / batch_size
        return result

class F1_score(object):
    def __init__(self,num_classes = None):
        self.labels = None
        if num_classes:
            self.labels = [i for i in range(num_classes)]
    def __call__(self,best_path,target):
        y_pred = best_path.contiguous().view(1, -1).squeeze().cpu()
        y_true = target.contiguous().view(1, -1).squeeze().cpu()
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        f1 = f1_score(y_true, y_pred, labels=self.labels, average="macro")
        correct = np.sum((y_true==y_pred).astype(int))
        acc = correct/y_pred.shape[0]
        return (acc, f1)

class Classification_Report(object):
    def __init__(self):
        pass
    def __call__(self,y_pred,y_true):
        _, y_pred = torch.max(y_pred.data, 1)
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        classify_report = classification_report(y_true, y_pred)
        print('\n\nclassify_report:\n', classify_report)


# 实体得分情况
class Entity_Score(object):
    def __init__(self,id_to_label):
        self.id_to_label = id_to_label
        self._reset()

    def _reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def _compute(self,origin,found,right):
        recall =0 if origin ==0 else (right / origin)
        precision= 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision ==0 else (2 * precision* recall) / (precision + recall)
        return recall,precision,f1

    def result(self):
        origin_counter = Counter([x['type'] for x in self.origins])
        found_counter = Counter([x['type'] for x in self.founds])
        right_counter = Counter([x['type'] for x in self.rights])
        for type,count in origin_counter.items():
            origin = count
            found = found_counter.get(type,0)
            right = right_counter.get(type,0)
            recall,precision,f1 = self._compute(origin,found,right)
            print("Type: %s - precision: %.4f - recall: %.4f - f1: %.4f"%(type,recall,precision,f1))

    def update(self,label_paths,pred_paths):
        '''
        :param label_paths: [[2,3,4,5,6,5,8,8,8,8,8,4],[2,3,7,7,7,8,8,8,8]]
        :param pred_paths: [[2,3,4,5,6,5,8,8,8,8,8,4],[2,3,7,7,7,8,8,8,8]]
        :return:
        '''
        for label_path,pre_path in zip(label_paths,pred_paths):
            label_entities = get_entity(label_path,self.id_to_label)
            pre_entities = get_entity(pre_path, self.id_to_label)
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])




