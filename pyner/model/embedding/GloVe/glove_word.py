#encoding:utf-8
import os
from os import path
def main():
    data_dir = '../../../dataset/raw'
    filename = path.sep.join([data_dir,'source_BIO_2014_cropus.txt'])
    with open('glove_word_vec.txt', 'w') as fw:
        with open(filename,'r') as fr:
            for i,line in enumerate(fr):
                line = line.strip('\n')
                fw.write(line+'\n')

if __name__ == "__main__":
    main()