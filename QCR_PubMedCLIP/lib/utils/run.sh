#!/bin/bash

####*******RAD dataset
DATAPATH=/home/nanaeilish/projects/mis/PubMedCLIP/QCR_PubMedCLIP/data
IMAGEPATH=/home/nanaeilish/projects/mis/PubMedCLIP/QCR_PubMedCLIP/data/data_rad/images
# python create_dictionary.py "$DATAPATH/data_rad" --dataset "rad" --trainfile "trainset.json" --testfile "testset.json"
# python create_label.py "$DATAPATH/data_rad/" --dataset "rad" --trainfile "trainset.json" --testfile "testset.json"
# python create_img2idx.py --train_path "$DATAPATH/data_rad/trainset.json" --val_path "$DATAPATH/data_rad/testset.json" --out_path "$DATAPATH/data_rad/imgid2idx.json"
# python ./create_resized_images.py ../../data/data_rad/imgid2idx.json $IMAGEPATH 84 ../../data/data_rad/images84x84.pkl 1
# python ./create_resized_images.py ../../data/data_rad/imgid2idx.json $IMAGEPATH 128 ../../data/data_rad/images128x128.pkl 1
python ./create_resized_images.py ../../data/data_rad/imgid2idx.json $IMAGEPATH 250 ../../data/data_rad/images250x250.pkl 3
# python ./create_resized_images.py ../../data/data_rad/imgid2idx.json $IMAGEPATH 288 ../../data/data_rad/images288x288.pkl 3


####*****SLAKE dataset

# python create_dictionary.py "path/to/data/data_slake"
# python create_label.py "path/to/data/data_slake/"
# python ./create_img2idx.py ../../data/data_slake/train.json ../../data/data_slake/test.json ../../data/data_slake/imgid2idx.json
# python ./create_resized_images.py ../../data/data_slake/imgid2idx.json path/to/Slake/imgs/ 84 ../../data/data_slake/images84x84.pkl 1
# python ./create_resized_images.py ../../data/data_slake/imgid2idx.json path/to/Slake/imgs/ 128 ../../data/data_slake/images128x128.pkl 1
# python ./create_resized_images.py ../../data/data_slake/imgid2idx.json path/to/Slake/imgs/ 250 ../../data/data_slake/images250x250.pkl 3
# python ./create_resized_images.py ../../data/data_slake/imgid2idx.json path/to/Slake/imgs/ 288 ../../data/data_slake/images288x288.pkl 3
