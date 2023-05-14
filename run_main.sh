
outdir=output_images
date=$(date '+%Y-%m-%d-%H-%M')
echo $date
outdir=$outdir/$date
mkdir -p $outdir
python main.py -t HEAD_CT -o $outdir --trainset trainset.json --testset testset.json
python main.py -t ABD_CT -o $outdir --trainset trainset.json --testset testset.json
python main.py -t CHEST_X-Ray -o $outdir --trainset trainset.json --testset testset.json
python main.py -t HEAD_MRI -o $outdir --trainset trainset.json --testset testset.json

# move the results to the directory
python collect_all_imgs.py  --indir $outdir