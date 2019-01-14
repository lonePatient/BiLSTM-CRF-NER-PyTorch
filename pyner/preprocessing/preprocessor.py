#encoding:utf-8
import re
import spacy
from nltk import word_tokenize
class Preprocessor(object):
    def __init__(self):
        pass
    def known_contractions(self,embed,contraction_mapping):
        known = []
        for contract in contraction_mapping:
            if contract in embed:
                known.append(contract)
        return known

    # 清洗特特殊符号
    def clean_contractions(self,text, mapping):
        specials = ["’", "‘", "´", "`"]
        for s in specials:
            text = text.replace(s, "'")
        text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
        return text

    def correct_spelling(self,x, dic):
        for word in dic.keys():
            x = x.replace(word, dic[word])
        return x

    def is_lower(self,text):
        return text.lower()

    def unknown_punct(self,embed, punct):
        unknown = ''
        for p in punct:
            if p not in embed:
                unknown += p
                unknown += ' '
        return unknown

    def clean_special_chars(self,text, punct, mapping):
        for p in mapping:
            text = text.replace(p, mapping[p])

        for p in punct:
            text = text.replace(p, f'{p}')

        specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '',
                    'है': ''}  # Other special characters that I have to deal with in last
        for s in specials:
            text = text.replace(s, specials[s])
        return text

    def add_lower(self,embedding, vocab):
        count = 0
        for word in vocab:
            if word in embedding and word.lower() not in embedding:
                embedding[word.lower()] = embedding[word]
                count += 1
        print(f"Added {count} words to embedding")

    def clean_numbers(self,x):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
        return x

    def __call__(self, sentence):
        # TorchText returns a list of words instead of a normal sentence.
        # First, create the sentence again. Then, do preprocess. Finally, return the preprocessed sentence as list
        # of words
        x = self.is_lower(sentence)
        x = self.clean_numbers(x)

        if x is None:
            x = '__##__'
        return x
