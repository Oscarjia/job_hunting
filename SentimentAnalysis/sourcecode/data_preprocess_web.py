

import argparse
import csv
import json
import re
from collections import Counter

import jieba


def replace_dish(content):
    return re.sub("【.{5,20}】","<dish>",content)

def normalize_num(words):
    '''Normalize numbers
    for example: 123 -> 100,  3934 -> 3000
    '''
    tokens = []
    for w in words:
        try:
            ww = w
            num = int(float(ww))
            if len(ww) < 2:
                tokens.append(ww)
            else:
                num = int(ww[0]) * (10**(len(str(num))-1))
                tokens.append(str(num))
        except:
            tokens.append(w)
    return tokens

def tokenize(content):
    content = content.replace("\u0006",'').replace("\u0005",'').replace("\u0007",'')
    tokens = []
    content = content.lower()
    # 去除重复字符
    content = re.sub('~+','~',content)
    content = re.sub('～+','～',content)
    content = re.sub('(\n)+','\n',content)
    for para in content.split('\n'):
        para_tokens = []
        words = list(jieba.cut(para))
        words = normalize_num(words)
        para_tokens.extend(words)
        para_tokens.append('<para>')
        tokens.append(' '.join(para_tokens))
    content = " ".join(tokens)
    content = re.sub('\s+',' ',content)
    content = re.sub('(<para> )+','<para> ',content)
    content = re.sub('(- )+','- ',content)    
    content = re.sub('(= )+','= ',content)
    content = re.sub('(\. )+','. ',content).strip()
    content = replace_dish(content)
    if content.endswith("<para>"):
        content = content[:-7]
    return content

def create_vocab(data, vocab_file, vocab_size):
    print("# Start to create vocab ...")
    words = Counter()
    for item in data:
        words.update(item['content'].split())
    special_tokens = ['<unk>','<sos>','<eos>']
    with open(vocab_file,'w') as f:
        for w in special_tokens:
            f.write(w + '\n')
        for w,_ in words.most_common(vocab_size-len(special_tokens)):
            f.write(w + '\n')
    print("# Created vocab file {0} with vocab size {1}".format(vocab_file,vocab_size))

def process_data(output_file, labels, sentence):
    item = {}
    with open(output_file[0],'w') as f:
        item['id'] = 0
        item['content'] = sentence
        for label in labels:
            item[label] = ""
        content = tokenize(item['content'].strip()[1:-1])
        item['content'] = content
        f.write(json.dumps(item,ensure_ascii=False)+'\n')



