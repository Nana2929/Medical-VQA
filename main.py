import re
from typing import List, Dict
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import preprocessing as pp

DATA_dir = Path('./QCR_PubMedCLIP/data/data_rad')
IMAGES_DIR = Path(DATA_dir / 'images/')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--target', type=str, default='HEAD_CT')
    parser.add_argument('--trainset', type=str, default='trainset.json')
    parser.add_argument('--testset', type=str, default='testset.json')
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default="./outputs/")

    args = parser.parse_args()

    return args


def load_image(img_name: str):
    img = cv2.imread(str(IMAGES_DIR / img_name), cv2.CV_8UC1)
    # img = cv2.imread(str(IMAGES_DIR / img_name))
    return img_name, img


def display_image(source, start, nrow=3, ncol=3):
    if not nrow or not ncol:
        raise ValueError('nrow and ncol must be positive integers')

    for i in range(nrow * ncol):
        item = source.iloc[start + i]
        img = cv2.imread(str(IMAGES_DIR / item['image_name']))
        id = ''.join(re.findall(r'\d+', item['image_name']))
        title = id + '\n' + item['question'] + '\n' + item['answer']
        scale = abs(nrow - 1) * 0.3, abs(ncol - 1) * 0.8

        plt.subplot(nrow, ncol, i + 1)
        plt.subplots_adjust(hspace=scale[0], wspace=scale[1])
        plt.imshow(img)
        plt.title(title, fontsize=8)
        plt.xticks([])
        plt.yticks([])

    plt.show()

def get_target_subset(source, target: str)-> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    source : List[Dict]
        train or test split, in json dict format
    target : str
        target modality, if 'HEAD', specify image organ with 'HEAD_CT' or 'HEAD_MRI'

    Returns
    -------
    List[Dict]: subset of source matching target modality
    """
    source['organ_mod'] = source['image_organ'] + '_' + source['modality']
    source = source[source['organ_mod'] == target]
    print(f'Unique organ_mods {target}: ', source['image_name'].nunique())

    return source


# timer decorator
def timer(func):
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f'Elapsed time: {end - start:.2f} seconds')

    return wrapper
@timer
def main():
    args = parse_args()
    target = args.target
    output_dir = args.output_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    trainset = pd.read_json(DATA_dir / args.trainset)
    testset = pd.read_json(DATA_dir / args.testset)

    trainset = get_target_subset(trainset, target)
    testset = get_target_subset(testset, target)
    # print length
    # set
    trainset = set(trainset['image_name'])
    testset = set(testset['image_name'])
    # print('unique train images: ', len(set(trainset)))
    # print('unique test images: ', len(set(testset)))
    train_img_set = [load_image(n) for n in trainset]
    test_img_set = [load_image(n) for n in testset]
    all_img_set = train_img_set + test_img_set
    pp.pipeline(all_img_set, target, out_dir=output_dir)


if __name__ == '__main__':
    main()
# output dir: /home/nanaeilish/projects/mis/Medical-VQA/QCR_PubMedCLIP/data/data_rad/preprocessed_images