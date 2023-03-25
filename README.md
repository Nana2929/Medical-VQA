
# PubMedCLIP
By far (2023/03/24) the sota method for VQA-Rad dataset.

[Arxiv paper: Does CLIP Benefit Visual Question Answering in the
Medical Domain as Much as it Does in the General Domain?](https://arxiv.org/pdf/2112.13906.pdf)

Dataset for VQA-Rad, see [Awenbocc/med-vqa](https://github.com/Awenbocc/med-vqa) (by checking issue QQ).
## 1. Try fine-tuning with `roco`:

Can't find the `/train/radiologytraindata.csv` in ROCO dataset repo.


## 2. Try training the data on `VQA-RAD`


According to the original paper,
> Our goal is to investigate the effect of using PubMedCLIP as a pre-trained visual encoder in MedVQA models. VQA in
this work is considered as a classification problem, where the objective is to find a mapping function f that maps an
image–question pair ($vi$, $qi$) to the natural language answer $ai$.
It looks like there are 2 pipelines for constructing the model pipeline:
    (a) MEVF: Overcoming Data Limitation in Medical Visual Question Answering
    (b) QCR: Medical Visual Question Answering via Conditional Reasoning [paper](https://dl.acm.org/doi/abs/10.1145/3394171.3413761?casa_token=E_IrwKfXPEMAAAAA:IC1Epmj0HbdWYzZWUfPpjbBJuMuL-iTdGbe1kVr5UQ4iVvfTgN_mgDBBEjyhqNBzRanKKlzyVQ)

**QCR (PubMedClip-RN50+AE)** acheives the  best accuracy.

### Bugfixes:
1. 我不確定哪裡可以下載 `imgid2idx.json` 的檔案。已經修好（`run.sh` 內忘記執行）
```
# error message that does not seem to interfere: lib/utils/run.sh
Num of answers that appear >= 0 times: 557
Num of open answers that appear >= 0 times: 515
Num of close answers that appear >= 0 times: 72
found /home/nanaeilish/projects/mis/PubMedCLIP/QCR_PubMedCLIP/data/data_rad/cache/trainval_ans2label.pkl
Traceback (most recent call last):
  File "create_label.py", line 329, in <module>
    compute_target(train_qa_pairs, total_ans2label, intersection, 'train', img_col, data) #dump train target to .pkl {question,image_name,labels,scores}
NameError: name 'intersection' is not defined
```
2. 使用 `lib/utils/run.sh` 後產出的中間產物，會導致type_classifier checkpoint loading 錯誤。
[issue with VQA-RAD training](https://github.com/sarahESL/PubMedCLIP/issues/9)
因此根據作者建議將 Awenbocc/med-vqa 的 `data` 資料夾複製到 `QCR_PubMedCLIP/data/data_rad` 下。

3. `clip` 的下載要按照：[openai/clip](https://github.com/openai/CLIP#usage)。
4. 留意 cuda driver 版本問題。
5. `QCR_PubMedCLIP/lib/utils` 資料夾下的 `run.sh` 裡面只要執行 create_resized_images 的 scripts 就好，其他的檔案用 `Awenbocc/med-vqa` 的。
6. 可以用這個指令確認 $PWD 底下的.jpg images 數量：`find . -name "*.jpg" | wc -l`

## 2.1 Training log

```
2023-03-25 15:41:21,394 INFO     [Validate] Val_Acc:67.184036%  |  Open_ACC:46.408840%   |  Close_ACC:81.111107%
2023-03-25 15:41:22,982 INFO     [Result] The best acc is 67.184036% at epoch 22
```
## 2.2 Pretrained checkpoints saved at:

`/home/nanaeilish/projects/mis/PubMedCLIP/QCR_PubMedCLIP/output/qcr/pubmedclipRN50_ae`
## 3. Exporting the prediction on `VQA-RAD`
## 4. Use the prediction and the data itself for data EDA