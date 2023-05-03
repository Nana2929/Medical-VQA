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
question_closed = dict()
question_open = dict()
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
                write_closed.writerow([data['image_name'],data['question'], data['answer']])
                if data['image_name'] not in question_closed :
                    question_closed[data['image_name']] = list()

                question_closed[data['image_name']].append([data['question'], data['answer']])
            else:
                write_opened.writerow([data['image_name'],data['question'], data['answer']])
                if data['image_name'] not in question_open :
                    question_open[data['image_name']] = list()
                question_open[data['image_name']].append([data['question'], data['answer']])
                
            a.append(data)
    
print(len(a))


traincsvfile.close()

for data in testset.iloc():
    if data['image_name'] in img_list:
        if data['image_organ']== 'CHEST':
            testwriter.writerow([data['image_name'], data['question'], data['answer']])
            if data['answer_type'] == 'CLOSED':
                write_closed.writerow([data['image_name'],data['question'], data['answer']])
                if data['image_name'] not in question_closed :
                    question_closed[data['image_name']] = list()
                question_closed[data['image_name']].append([data['question'], data['answer']])
            else:
                write_opened.writerow([data['image_name'],data['question'], data['answer']])
                if data['image_name'] not in question_open :
                    question_open[data['image_name']] = list()
                question_open[data['image_name']].append([data['question'], data['answer']])
            b.append(data)

    
print(len(b))

testcsvfile.close()
closed.close()
opened.close()

write_image_closed = open("QuestionClosed.txt", 'w', encoding='UTF8', newline='')
write_image_opened = open("QuestionOpened.txt", 'w', encoding='UTF8', newline='')

for key in question_closed:
    write_image_closed.write(key + " "+ str(len(question_closed[key]))+ "\n")
    print(key, len(question_closed[key]))

    for item in question_closed[key]:
        write_image_closed.write(str(item) + "\n")
    write_image_closed.write("\n")
    write_image_closed.write("============================================================\n")



for key in question_open:
    write_image_opened.write(key + " "+ str(len(question_open[key]))+ "\n")
    for item in question_open[key]:
        write_image_opened.write(str(item) + "\n")
    write_image_opened.write("\n")
    write_image_opened.write("============================================================\n")

print("question image  in opened" , len(question_open))


print("question image  in closed" , len(question_closed))


# head CT
# head MRI
# chest X-Ray
# abdominal CT