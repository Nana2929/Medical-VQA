
# PubMedCLIP
By far (2023/03/24) the sota method for `VQA-Rad` dataset.

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
training 指令：
`python main.py --cfg configs/qcr_pubmedclipRN50_ae_rad_16batchsize_withtfidf_nondeterministic.yaml`

### Bugfixes
1. 我不確定哪裡可以下載 `imgid2idx.json` 的檔案。已經修好（`run.sh` 內忘記執行）。
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

## 2.1 Training log & Tensorboards (2023/03/25)

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
QCR (PubMedClip-RN50+AE) acheives the  best accuracy.
Bugfixes:
1. 我不確定哪裡可以下載 `imgid2idx.json` 的檔案。已經修好（`run.sh` 內忘記執行）
```
2023-03-25 15:41:21,394 INFO     [Validate] Val_Acc:67.184036%  |  Open_ACC:46.408840%   |  Close_ACC:81.111107%
2023-03-25 15:41:22,982 INFO     [Result] The best acc is 67.184036% at epoch 22

2023-03-25 18:58:54,497 INFO     -------[Epoch]:199-------
2023-03-25 18:58:54,497 INFO     [Train] Loss:0.000347 , Train_Acc:99.184074%
2023-03-25 18:58:54,497 INFO     [Train] Loss_Open:0.000015 , Loss_Close:0.000026%
2023-03-25 18:58:56,648 INFO     [Validate] Val_Acc:69.844788%  |  Open_ACC:55.801105%   |  Close_ACC:79.259254%
2023-03-25 18:58:56,649 INFO     [Result] The best acc is 71.618622% at epoch 62 # paper: 72.1%
```
Train Acc 大概在 25 epochs 時就幾乎達到最高峰。
Closed 的 VAcc 很不穩定，有個 2 ~ 3 % 之間的劇烈下跌。
Open 的 VAcc 整體呈現上升趨勢。
整體而言和 Paper 的數據有約莫 0.5% 的差距（較低）。

![](QCR_PubMedCLIP/output/qcr/pubmedclipRN50_ae/roco/VQARAD/QCR.CLIPRN50.AE.ROCO.VQARAD.16batchsize.200epoch.withTFIDF.nondeterministic/imgs/2023_0405_訓練結果tensorboard.png)

## 2.2 Pretrained checkpoints saved at:
`/home/nanaeilish/projects/mis/PubMedCLIP/QCR_PubMedCLIP/output/qcr/pubmedclipRN50_ae`
## 3. Exporting the prediction on `VQA-RAD`

testing 指令：
`python main.py --test True --cfg configs/qcr_pubmedclipRN50_ae_rad_16batchsize_withtfidf_nondeterministic.yaml`

Add these (or at least your configured paths) into the config file `--cfg`.

```
MODEL_FILE: "PubMedCLIP/QCR_PubMedCLIP/output/qcr/pubmedclipRN50_ae/roco/VQARAD/QCR.CLIPRN50.AE.ROCO.VQARAD.16batchsize.200epoch.withTFIDF.nondeterministic/62_best.pth"
  RESULT_DIR: "PubMedCLIP/QCR_PubMedCLIP/output/qcr/pubmedclipRN50_ae/roco/VQARAD/QCR.CLIPRN50.AE.ROCO.VQARAD.16batchsize.200epoch.withTFIDF.nondeterministic/results"
```
### 成績
`PubMedCLIP/QCR_PubMedCLIP/output/qcr/pubmedclipRN50_ae/roco/VQARAD/QCR.CLIPRN50.AE.ROCO.VQARAD.16batchsize.200epoch.withTFIDF.nondeterministic/medVQA.log` 中的 validation best acc is `71.618622%` at epoch 62，最後 test 使用 epoch 62 的checkpoint的成績是 `71.175163%`。根據 `/QCR_PubMedCLIP/main.py`，test mode 下依然使用 val_loader，因此不確定為何會有分數差。
```
451 179.0 272.0   # 全部，Open，Close 的數量
[Validate] Val_Acc:71.175163%  |  Open_ACC:56.424580%   |  Close_ACC:80.882355%
```

> 2023/04/06 Issue
在寫成 testfile 時他的 Predicted_answer 都只寫出 closed_logits
是 tensor 的形式，沒有轉換成文字，要自己 decode。[Hackmd Notes: How to get the inference data](https://hackmd.io/@NanaEilish727/pmclip)


## 4. Use the prediction and the data itself for data EDA

1. Data Example。`question_type` 和 `phrase_type` 的數量。
2. `question_type` 和 `phrase_type` level 的 EDA（類別正確與錯誤率）。





## Todo:
0. 完整的設定檔理解（data preproc 到底做到哪，有 resize 嗎，有旋轉嗎）
1. xai 套件如 LIME
2. medical preprocessing （傅立葉轉換、輪廓清晰 filter）工具包試用

