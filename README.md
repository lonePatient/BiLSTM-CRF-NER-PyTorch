## PyTorch solution of NER task Using BiLSTM-CRF model.

This repo contains a PyTorch implementation of a BiLSTM-CRF model for named entity recognition task.

## Structure of the code

At the root of the project, you will see:

```text
├── pyner
|  └── callback
|  |  └── lrscheduler.py　　
|  |  └── trainingmonitor.py　
|  |  └── ...
|  └── config
|  |  └── basic_config.py #a configuration file for storing model parameters
|  └── dataset　　　
|  └── io　　　　
|  |  └── data_loader.py　　
|  |  └── data_transformer.py　　
|  └── model
|  |  └── embedding
|  |  └── layers
|  |  └── nn
|  └── output #save the ouput of model
|  └── preprocessing #text preprocessing 
|  └── train #used for training a model
|  |  └── trainer.py 
|  |  └── ...
|  └── utils # a set of utility functions
|  └── test
├── test_predict.py
├── train_bilstm_crf.py
├── train_word2vec.py
```
## Dependencies

- csv
- tqdm
- numpy
- pickle
- scikit-learn
- PyTorch 1.0
- matplotlib

## How to use the code

1. Download the `source_BIO_2014_cropus.txt` from [BaiduPan](https://pan.baidu.com/s/1LDwQjoj7qc-HT9qwhJ3rcA)(password: 1fa3) and place it into the `/pyner/dataset/raW` directory.
2. Modify configuration information in `pyner/config/basic_config.py`(the path of data,...).
3. run `python train_bilstm_crf.py` ．
4. run `python test_predict.py` ．


## Result

```text
----------- Train entity score:
Type: LOC - precision: 0.9043 - recall: 0.9089 - f1: 0.9066
Type: PER - precision: 0.8925 - recall: 0.9215 - f1: 0.9068
Type: ORG - precision: 0.8279 - recall: 0.9016 - f1: 0.8632
Type: T - precision: 0.9408 - recall: 0.9462 - f1: 0.9435
----------- valid entity score:
Type: T - precision: 0.9579 - recall: 0.9558 - f1: 0.9568
Type: PER - precision: 0.9058 - recall: 0.9205 - f1: 0.9131
```

### training Figure

![]( https://lonepatient-1257945978.cos.ap-chengdu.myqcloud.com/20190225223107.png)
