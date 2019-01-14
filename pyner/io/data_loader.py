#encoding:utf-8
import json
from torchtext import data

# 定义Dataset
# 原始数据我们保存成一行一条json格式
class CreateDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 text_field,
                 label_field,
                 x_var,y_var,
                 skip_header,
                 is_train_mode,
                 ):
        fields = [(x_var, text_field), (y_var, label_field)]
        examples = []
        if is_train_mode:
            with open(data_path,'r') as fr:
                for i,line in enumerate(fr):
                    # 第一行一般为列名
                    if i == 0 and skip_header:
                        continue
                    df = json.loads(line)
                    sentence = df[x_var]
                    label = df[y_var]
                    examples.append(data.Example.fromlist([sentence, label], fields))
        else:
            with open(data_path,'r') as fr:
                for i,line in enumerate(fr):
                    if i == 0 and skip_header:
                        continue
                    df = json.loads(line)
                    sentence = df[x_var]
                    label = df[y_var]
                    examples.append(data.Example.fromlist([sentence,label], fields))
        super(CreateDataset,self).__init__(examples,fields)

class DataLoader(object):
    def __init__(self,
                 data_path,
                 batch_size,
                 logger,
                 x_var,
                 y_var,
                 skip_header = False,
                 is_train_mode = True,
                 max_sentence_length = None,
                 device = 'cpu'
                 ):

        self.logger           = logger
        self.device           = device
        self.batch_size       = batch_size
        self.data_path        = data_path
        self.x_var            = x_var
        self.y_var            = y_var
        self.is_train_mode    = is_train_mode
        self.skip_header      = skip_header
        self.max_sentence_len = max_sentence_length

    def make_iter(self):

        TEXT = data.Field(sequential      = True,
                          use_vocab       = False,
                          tokenize        = lambda x: [int(c) for c in x.split()],  # 如果加载进来的是已经转成id的文本, 此处必须将字符串转换成整型
                          fix_length      = self.max_sentence_len, # 如需静态padding,则设置fix_length, 但要注意要大于文本最大长度
                          pad_token       = 0,  # 这里需要注意，与DataTransformer的pad形式对应
                          batch_first     = True,
                          eos_token       = None,
                          init_token      = None,
                          include_lengths = True
                          )

        LABEL = data.Field(sequential      = True,
                           use_vocab       = False,
                           tokenize        = lambda x: [int(c)+1 for c in x.split()],
                           batch_first     = True,
                           fix_length      = self.max_sentence_len,
                           eos_token       = None,
                           init_token      = None,
                           include_lengths = False,
                           pad_token       = 0     # 这里需要注意：我们使用0进行填充
                           )

        dataset = CreateDataset(data_path = self.data_path,
                               text_field  = TEXT,
                               label_field = LABEL,
                               x_var  = self.x_var,
                               y_var  = self.y_var,
                               skip_header = self.skip_header,
                               is_train_mode = self.is_train_mode
                              )
        # # 构建Iterator
        # # 在 test_iter, shuffle, sort, repeat一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序
        # # 如果输入变长序列，sort_within_batch需要设置成true，使每个batch内数据按照sort_key降序进行排序
        if self.is_train_mode:
            data_iter = data.BucketIterator(dataset = dataset,
                                            batch_size = self.batch_size,
                                            shuffle = True,
                                            repeat=False,
                                            sort_within_batch = True,
                                            sort_key = lambda x:len(getattr(x,self.x_var)),
                                            device = self.device)
        else:
            data_iter = data.Iterator(dataset = dataset,
                                      batch_size = self.batch_size,
                                      shuffle = False,
                                      sort = False,
                                      repeat = False,
                                      device = self.device)
        return BatchWrapper(data_iter, x_var=self.x_var, y_var=self.y_var)

class BatchWrapper(object):
    """对batch做个包装，方便调用，可选择性使用"""
    def __init__(self, dl, x_var, y_var):
        self.dl, self.x_var, self.y_var = dl, x_var, y_var

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)
            target = getattr(batch, self.y_var)
            source = x[0]
            length = x[1]
            yield (source, target, length)
    def __len__(self):
        return len(self.dl)
