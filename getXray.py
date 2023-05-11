import os
import csv
import cv2
import json
from typing import Union, List, Dict 
import pandas as pd
folder_path = "data/images/"
output_path = "data"
train_test_list = ['Train', 'Test']
json_input_path = ["data/trainset.json","data/testset.json"]
img_list = os.listdir(folder_path)
img_list.sort(key = lambda x : int(x[6:-4]))

# os.mkdir("./data/Xray_closed")
# os.mkdir("./data/Xray_opened")
# os.mkdir("./data/Xray_closed/train")
# os.mkdir("./data/Xray_opened/train")
# os.mkdir("./data/Xray_closed/test")
# os.mkdir("./data/Xray_opened/test")
#    qid       image_name image_organ      answer answer_type question_type                                       question phrase_type
# data to pandas dataframe
def to_dataframe(data: List[Dict]):
    # data: List of dicts 
    # to data frame
    df = pd.DataFrame(data)
    return df

# # closed open write title
# csv_list = ['XrayClosedAnswer.csv', 'XrayOpenAnswer.csv']
# for csvfile in csv_list:
#     if os.path.isfile(csvfile):
#         os.remove(csvfile)
#     file = open(csvfile, 'w', encoding='UTF8', newline='')
#     write =csv.writer(file)
#     write.writerow(['image_name', 'question','question_type','answer'])


# question_type = dict()

# question_type_file = open("QuestionType.txt", 'w', encoding='UTF8', newline='')

# train_test_question = dict()
# train_test_question['Train'] = dict()
# train_test_question['Test'] = dict()

# # collect answer type
# answer_type = dict()
# answer_type["OPEN"] = dict()
# answer_type["CLOSED"] = dict()
# def answer_type_write(data,close_open,csv_list):
#     if data['image_name'] not in  answer_type[close_open]:
#         answer_type[close_open][data['image_name']] = list()
#     answer_type[close_open][data['image_name']].append([data['question'], data['question_type'], data['answer']])

# # write csv file split closed and open
# def closed_open_write(data):
#     if data['answer_type'] == 'CLOSED':
#         file = open(csv_list[0], 'a', encoding='UTF8', newline='')
#         write_file =csv.writer(file)
#         write_file.writerow([data['image_name'],data['question'], data['question_type'], data['answer']])
#         answer_type_write(data,"CLOSED",csv_list)
#         file.close()

#     else:
#         file = open(csv_list[1], 'a', encoding='UTF8', newline='')
#         write_file =csv.writer(file)
#         write_file.writerow([data['image_name'],data['question'], data['question_type'], data['answer']])
#         answer_type_write(data,"OPEN",csv_list)
#         file.close()


# # write csv file split train and test
# def train_test(dataset,output_write,organ = 'CHEST'): # path "./data/Xray_closed/train/"
#     for data in dataset.iloc() :
#         if data['image_name'] in img_list:
#             if data['image_organ']== organ:
#                 if data['question_type'] not in question_type:
#                     question_type[data['question_type']] = list()
#                 question_type[data['question_type']].append([data['question'], data['answer'], data['answer_type'], data['image_name']])
                
#                 output_write.writerow([data['image_name'], data['question'], data['question_type'], data['answer']])
#                 closed_open_write(data)                   
#     # print(question_type.keys())

# write csv file split train and test
q_type = dict()
if not os.path.isdir("./Xray"):
    os.mkdir("./Xray")
for it in range(len(train_test_list)):
    with open(json_input_path[it]) as f:
        jsonset = json.load(f)
        jsonset = to_dataframe(jsonset)
        jsonset = jsonset.drop(["qid"], axis=1)
        mask1 = jsonset['image_organ'] == 'CHEST'
        mask2 = jsonset['image_name'].isin(img_list)
        filter = jsonset[mask1 & mask2]
        print(train_test_list[it] , len(filter))
        image_name = filter.groupby('image_name')
        image_name = image_name[["question","answer","answer_type","question_type"]]

        with open(train_test_list[it]+".txt", 'w', encoding='UTF8', newline='') as image_name_file:
            image_name_file.write("image_name,question,answer,answer_type,question_type")
            image_name_file.write('\n')

            for g in image_name:
                image_name_file.write(g[0].split('.')[0]+":"+str(len(g[1]))+"\n")
                image_name_file.write('\n')
                image_name_file.write(g[1].to_csv(index=False, header=False)+'\n')
                image_name_file.write('=====================================\n')
                cv2.imwrite('./Xray/'+g[0], cv2.imread(folder_path+g[0]))



        open_closed = filter.groupby('answer_type',sort ='image_name' )
        open_closed = open_closed[["image_name","question","answer","question_type"]]

        for g in open_closed:
                with open(train_test_list[it]+g[0]+".csv", 'w', encoding='UTF8', newline='') as open_close_file:
                    open_close_file.write("image_name,question,answer,question_type")
                    open_close_file.write('\n')
                    open_close_file.write(g[1].to_csv(index=False, header=False)+'\n')
        
        question_type = filter.groupby('question_type',sort ='image_name' )
        question_type = question_type[["image_name","question","answer","answer_type"]]

        with open(train_test_list[it]+"QuestionType"+".txt", 'w', encoding='UTF8', newline='') as question_type_file:
            question_type_file.write("question_type,question,answer,answer_type")
            question_type_file.write('\n')

            for g in question_type:
                if g[0] not in q_type:
                    q_type[g[0]] = dict()

                question_type_file.write(g[0]+":"+str(len(g[1]))+"\n")
                question_type_file.write('\n')
                name = g[1].groupby('image_name')
                name = name[["question","answer","answer_type"]]


                
                for n in name:
                    if n[0].split('.')[0] not in q_type[g[0]]:
                        q_type[g[0]][n[0].split('.')[0]] = list()
                    for it in n[1].itertuples():

                        q_type[g[0]][n[0].split('.')[0]].append(str(it.question)+","+str(it.answer)+","+str(it.answer_type))
                    question_type_file.write(n[0].split('.')[0]+":"+str(len(n[1]))+"\n")
                    question_type_file.write('\n')
                    question_type_file.write(n[1].to_csv(index=False, header=False))
                    question_type_file.write('=====================================\n')
                

                question_type_file.write('=====================================\n')


                



                # print("open_close",g[0], len(g[1]))
        # open_closed = open_closed[["question","answer","answer_type","question_type"]]
        # for g in open_closed:
        #     with open(train_test_list[it]+)

        # for name, dataset in image_name:
        #     if name not in train_test_question[train_test_list[it]]:
        #         train_test_question[train_test_list[it]][name] = list()
        #     train_test_question[train_test_list[it]][name].append(dataset)
            # train_test(dataset,write)
        # jsonset = jsonset[fliter]
        
        # train_test(jsonset,write)

write = open("QuestionType.txt", 'w', encoding='UTF8', newline='')
for typ in q_type:
    total = sum([len(q_type[typ][image]) for image in q_type[typ]])
    write.write(typ+":"+str(total)+"\n")
    # write.write(typ+":"+str(len(q_type[typ]))+"\n")
    for image in q_type[typ]:
        write.write(image+":"+str(len(q_type[typ][image]))+"\n")
        for item in q_type[typ][image]:
            write.write(str(item)+"\n")
        write.write("=====================================\n")
write.close()




# # write image QA to txt file
# # def write_imageQA(answer_type):
# for data in answer_type :
#     write = open("Question"+data+".txt", 'w', encoding='UTF8', newline='')
#     write.write("images : " + str(len(answer_type[data])) + "\n")
#     write.write("\n")


#     for image in answer_type[data] :
#         imagename = image.split('.')[0]
#         write.write(imagename+ " : "+ str(len(answer_type[data][image]))+ "\n")
        
#         for item in answer_type[data][image]:
#             write.write(str(item) + "\n")
#         write.write("============================================================\n")
#     write.close()





    # for filename in img_list:





# head CT
# head MRI
# chest X-Ray
# abdominal CT