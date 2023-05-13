import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import preprocess as pp

DATA_dir = Path('./data/')
IMAGES_DIR = Path(DATA_dir / 'images/')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--target', type=str, default='HEAD')
    parser.add_argument('--trainset', type=str, default='trainset.json')
    parser.add_argument('--testset', type=str, default='testset.json')
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default="./outputs/")

    args = parser.parse_args()

    return args


def load_image(idx, source):
    item = source.iloc[idx]
    img_name = item['image_name']
    img = cv2.imread(str(IMAGES_DIR / img_name), cv2.CV_8UC1)
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


def main():
    args = parse_args()
    target = args.target
    output_dir = args.output_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    trainset = pd.read_json(DATA_dir / args.trainset)
    testset = pd.read_json(DATA_dir / args.testset)

    trainset = trainset[trainset['image_organ'] == target]
    testset = testset[testset['image_organ'] == target]

    train_img_set = [load_image(i, trainset) for i in range(len(trainset))]
    # test_img_set = [load_image(i, testset) for i in range(len(testset))]

    pp.pipeline(train_img_set, target, out_dir=output_dir)


if __name__ == '__main__':
    main()