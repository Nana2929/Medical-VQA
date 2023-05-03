import os
import csv
from typing import Union, List, Dict 
import pandas as pd
folder_path = "data/images/"
train_output = 'XrayTrain.csv'
test_output = 'XrayTest.csv'
json_train_path = "data/trainset.json"
json_test_path = "data/testset.json"
img_list = os.listdir(folder_path)
img_list.sort(key = lambda x : int(x[6:-4]))

def to_dataframe(data: List[Dict]):
    # data: List of dicts 
    # to data frame
    df = pd.DataFrame(data)
    return df

import json
with open(json_train_path, 'r') as f:
    trainset = json.load(f)

    trainset = to_dataframe(trainset)
    # trainset = trainset.set_index('image_name')
with open(json_test_path, 'r') as f:
    testset = json.load(f)

    testset = to_dataframe(testset)

a= list()
b= list()
question_closed = list()
question_open = list()
closed = open("XrayClosedAnswer.csv", 'w', encoding='UTF8', newline='')
write_closed =csv.writer(closed)
write_closed.writerow(['image_name', 'question','answer'])

opened = open("XrayOpenAnswer.csv", 'w', encoding='UTF8', newline='')
write_opened =csv.writer(opened)
write_opened.writerow(['image_name', 'question','answer'])

traincsvfile =  open(train_output, 'w', encoding='UTF8', newline='') 
trainwriter = csv.writer(traincsvfile)
trainwriter.writerow(['image_name', 'question','answer'])

testcsvfile =  open(test_output, 'w', encoding='UTF8', newline='')
testwriter = csv.writer(testcsvfile)
testwriter.writerow(['image_name', 'question','answer'])

    # for filename in img_list:
for data in trainset.iloc() :
    if data['image_name'] in img_list:
        if data['image_organ']== 'CHEST':
            trainwriter.writerow([data['image_name'], data['question'], data['answer']])
            if data['answer_type'] == 'CLOSED':
                write_closed.writerow([data['question'], data['answer']])
                question_closed.append([data['question'], data['answer']])
            else:
                write_opened.writerow([data['question'], data['answer']])
                question_open.append([data['question'], data['answer']])
            a.append(data)
    
print(len(a))

traincsvfile.close()

for data in testset.iloc():
    if data['image_name'] in img_list:
        if data['image_organ']== 'CHEST':
            testwriter.writerow([data['image_name'], data['question'], data['answer']])
            if data['answer_type'] == 'CLOSED':
                write_closed.writerow([data['question'], data['answer']])
                question_closed.append([data['question'], data['answer']])
            else:
                write_opened.writerow([data['question'], data['answer']])
                question_open.append([data['question'], data['answer']])
            b.append(data)
    
print(len(b))

testcsvfile.close()
closed.close()
opened.close()

# head CT
# head MRI
# chest X-Ray
# abdominal CT