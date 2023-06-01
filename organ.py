# python ==3.8
# pip install -U spacy
# pip install scispacy

# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bionlp13cg_md-0.5.1.tar.gz
import spacy
from spacy import displacy
from scispacy.abbreviation import AbbreviationDetector

# pip install nltk
import nltk 
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')  # 依 WordNet 字典將各詞還原成原形
nltk.download('punkt')  # 斷詞用


from typing import Union, List, Dict 
import pandas as pd
import numpy as np
import os

# #pip install pickle4
# import pickle4 as pickle

# nltk.download('punkt')
if os.path.exists('Type') == False:
    os.mkdir('Type')

testset_path = "data/testset.json"
trainset_path = "data/trainset.json"
# read json 
# # Load the model
nlp = spacy.load("en_ner_bionlp13cg_md")
nlp.add_pipe("abbreviation_detector")

def to_dataframe(data: List[Dict]):
    # data: List of dicts 
    # to data frame
    df = pd.DataFrame(data)
    return df

import json

def get_organ(json_path):
    dic = dict()

    with open(json_path, 'r') as f:
        trainset = json.load(f)
        trainset = to_dataframe(trainset)

    for item in trainset.iloc:
        doc = nlp(str(item['question']) +' '+ str(item['answer']))
        for ent in doc.ents:
            txt = nltk.word_tokenize(ent.text)

            # lower
            txt = [w.lower() for w in txt]
            lemmatizer = WordNetLemmatizer()  

            # lemmatize  
            txt = [lemmatizer.lemmatize(w) for w in txt]
            sentence = ' '.join(txt)
            label = str(ent.label_)
            if int(str(item['qid'])) not in dic:
                dic[int(str(item['qid']))] = list()
            dic[int(str(item['qid']))].append(sentence)
               
    return dic 

def write_json(dic, json_path):
    json_data = json.dumps(dic, indent=4)
    with open(json_path, "w") as outfile:
        outfile.write(json_data)

def write_pickle(dic, pickle_path):
    with open(pickle_path, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
organdict = dict()
traindic  = get_organ(trainset_path)
testdic = get_organ(testset_path)
organdict.update(traindic)
organdict.update(testdic)

write_json(organdict, "./Type/all.json")
write_json(traindic, "./Type/train.json")
write_json(testdic, "./Type/test.json")









