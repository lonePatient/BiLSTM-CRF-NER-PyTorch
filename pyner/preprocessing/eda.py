import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

'''
1. 简单查看数据分布情况
'''
def label_freq():
    sin = len(train_df[train_df["class"]==0])
    insin = len(train_df[train_df["class"]==1])
    persin = (sin/(sin+insin))*100
    perinsin = (insin/(sin+insin))*100
    print("# Sincere questions: {:,}({:.2f}%) and # Insincere questions: {:,}({:.2f}%)".format(sin,persin,insin,perinsin))
    # print("Sinsere:{}% Insincere: {}%".format(round(persin,2),round(perinsin,2)))
    print("# Test samples: {:,}({:.3f}% of train samples)".format(len(test_df),len(test_df)/len(train_df)))

if __name__ == '__main__':

    train_df = pd.read_csv('../../input/train_set.csv')
    test_df = pd.read_csv('../../input/test_set.csv')

    train_word = train_df['word'].values.tolist()
    train_label = train_df['class'].values
    test_word = test_df['word'].values.tolist()

    num_label = len(set(train_label))
    print(f'# of labels: {num_label}')

    train_word_len = [len(words.split()) for words in train_word]
    print('Training set')
    print(np.percentile(train_word_len, [0, 50, 80, 90, 95, 98, 100]))

    test_word_len = [len(words.split()) for words in test_word]
    print('Test set')
    print(np.percentile(test_word_len, [0, 50, 80, 90, 95, 98, 100]))

