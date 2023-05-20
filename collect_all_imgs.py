
from pathlib import Path
from collections import defaultdict
import shutil
import argparse
def main(args):
    preprocess_img_dir = args.indir
    organ_mods = ['HEAD_CT', 'HEAD_MRI', 'ABD_CT', 'CHEST_X-Ray']
    merged_dir = f'{preprocess_img_dir}/merged_final'
    # makedir
    Path(merged_dir).mkdir(parents=True, exist_ok=True)

    img_count = defaultdict(int)
    for organ_mod in organ_mods:
        om_dir = f'{preprocess_img_dir}/{organ_mod}'
        final_dir = f'{om_dir}/final'
        for img in Path(final_dir).iterdir():
            img_count[organ_mod] += 1
            img_name = img.name
            shutil.copy(img, f'{merged_dir}/{img_name}')

    print(f'Number of images: {img_count}')
    gold_img_counts = defaultdict(int)
    gold_img_dir = '/home/nanaeilish/projects/mis/Medical-VQA/QCR_PubMedCLIP/data/data_rad/images_classified/body_mod'
    for organ_mod in organ_mods:
        om_dir = f'{gold_img_dir}/{organ_mod}'
        for img in Path(om_dir).iterdir():
            gold_img_counts[organ_mod] += 1
            img_name = img.name

    print(f'Number of gold images: {gold_img_counts}')
    print(f'Total: {sum(img_count.values())}, {sum(gold_img_counts.values())}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='output_images/2023-05-14-18-15')
    args = parser.parse_args()
    main(args)
