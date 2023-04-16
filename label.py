import os
import csv
from typing import Union, List, Dict 
import pandas as pd
folder_path = "data/images/"
path = 'output3.csv'
json_path = "data/trainset.json"
img_list = os.listdir(folder_path)
img_list.sort(key = lambda x : int(x[6:-4]))

def to_dataframe(data: List[Dict]):
    # data: List of dicts 
    # to data frame
    df = pd.DataFrame(data)
    return df

import json
with open(json_path, 'r') as f:
    trainset = json.load(f)

    trainset = to_dataframe(trainset)
    # trainset = trainset.set_index('image_name')
    trainset = trainset[['image_name','image_organ']]
    trainset = trainset.drop_duplicates()
    trainset = trainset.set_index('image_name')

print(trainset)

with open(path, 'w', encoding='UTF8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['imageID', 'position','class'])
    
    for filename in img_list:
        if filename in trainset.index:
            if trainset.loc[filename]['image_organ'] == 'CHEST':
                writer.writerow([filename, trainset.loc[filename]['image_organ'], 'X-Ray'])
            elif trainset.loc[filename]['image_organ'] == 'ABD':
                writer.writerow([filename, trainset.loc[filename]['image_organ'], 'CT'])
            elif trainset.loc[filename]['image_organ'] == 'HEAD':
                writer.writerow([filename, trainset.loc[filename]['image_organ'], 'CT/MRI'])
            else:
                writer.writerow([filename, trainset.loc[filename]['image_organ'], 'none'])



# head CT
# head MRI
# chest X-Ray
# abdominal CT