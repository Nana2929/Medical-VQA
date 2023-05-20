from collections import defaultdict
from pathlib import Path
import json
import shutil

dataset_dir = Path('QCR_PubMedCLIP/data/data_rad')
trainset = dataset_dir/'trainset.json'
testset = dataset_dir/'testset.json'

mod_dict = defaultdict(set)
body_mod_dict = defaultdict(set)

output_dir = Path('QCR_PubMedCLIP/data/data_rad/images_classified')
mod_dir = output_dir/'mod'
body_mod_dir = output_dir/'body_mod'
# create directories
mod_dir.mkdir(parents=True, exist_ok=True)
body_mod_dir.mkdir(parents=True, exist_ok=True)


# iterate thru both train and test sets
for dataset in [trainset, testset]:
    with open(dataset, 'r') as f:
        data = json.load(f)
        for item in data:
            image_name = item['image_name']
            modality = item['modality']
            body_part = item['image_organ']
            body_mod = body_part + '_' + modality
            image_path = dataset_dir/'images'/image_name

            # create directories
            each_mod_dir = output_dir/'mod'/modality
            each_body_mod_dir = output_dir/'body_mod'/body_mod
            each_mod_dir.mkdir(parents=True, exist_ok=True)
            each_body_mod_dir.mkdir(parents=True, exist_ok=True)


            # copy the image to the corresponding directory
            if image_name not in mod_dict[modality]:
                shutil.copy(image_path, each_mod_dir)
                mod_dict[modality].add(image_name)
            if image_name not in body_mod_dict[body_mod]:
                shutil.copy(image_path, each_body_mod_dir)
                body_mod_dict[body_mod].add(image_name)
# output statistics
print('- modality statistics:')
for k, v in mod_dict.items():
    print('\t', k, len(v))
print('- body + mod statistics:')
for k, v in body_mod_dict.items():
    print('\t', k, len(v))