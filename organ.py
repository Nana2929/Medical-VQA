# python ==3.8
# pip install -U spacy
# python -m spacy download en_core_web_sm
# pip install scispacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_jnlpba_md-0.5.1.tar.gz
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_craft_md-0.5.1.tar.gz
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bionlp13cg_md-0.5.1.tar.gz
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz
import spacy
from spacy import displacy
from scispacy.abbreviation import AbbreviationDetector


import nltk 
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')  # 依 WordNet 字典將各詞還原成原形
nltk.download('punkt')  # 斷詞用

from typing import Union, List, Dict 
import pandas as pd
import numpy as np
# nltk.download('punkt')

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

def get_organ(organdict,json_path):
    dic = dict()

    with open(json_path, 'r') as f:
        trainset = json.load(f)

        trainset = to_dataframe(trainset)

    for item in trainset.iloc:

        doc = nlp(str(item['question']) +' '+ str(item['answer']))
        for ent in doc.ents:
            if ent.label_ == 'ORGAN':
                txt = nltk.word_tokenize(ent.text)
                # lower
                txt = [w.lower() for w in txt]
                lemmatizer = WordNetLemmatizer()    
                txt = [lemmatizer.lemmatize(w) for w in txt]
                sentence = ' '.join(txt)


                if sentence not in organdict:
                    organdict[sentence] = list()
                if sentence not in dic:
                    dic[sentence] = list()
                organdict[sentence].append(int(item['qid']))
                dic[sentence].append(int(item['qid']))
    return dic , organdict

def write_json(dic, json_path):
    json_data = json.dumps(dic)
    with open(json_path, "w") as outfile:
        outfile.write(json_data)

organdict = dict()
traindic , organdict = get_organ(organdict,trainset_path)
testdic,organdict = get_organ(organdict,testset_path)
write_json(organdict, "organdict.json")
write_json(traindic, "traindic.json")
write_json(testdic, "testdic.json")

print(organdict.keys())
print(len(organdict.keys()))

print(traindic.keys())
print(len(traindic.keys()))
print(testdic.keys())

print(len(testdic.keys()))




