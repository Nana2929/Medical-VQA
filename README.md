
# PubMedCLIP
By far (2023/03/24) the sota method for `VQA-Rad` dataset.

[Arxiv paper: Does CLIP Benefit Visual Question Answering in the
Medical Domain as Much as it Does in the General Domain?](https://arxiv.org/pdf/2112.13906.pdf)

Dataset for VQA-Rad, see [Awenbocc/med-vqa](https://github.com/Awenbocc/med-vqa) (by checking issue QQ).

## 各種 Visual Encoders 試用

- Consistent Configs:
  - QCR
  - pubmedclip
### Visual Encoders: RN50, RN50x4, ViT
- RN50
  ```
  Sat, 25 Mar 2023 16:26:49 INFO -------[Epoch]:62-------
  Sat, 25 Mar 2023 16:26:49 INFO [Train] Loss:0.000482 , Train_Acc:99.053528%
  Sat, 25 Mar 2023 16:26:49 INFO [Train] Loss_Open:0.000029 , Loss_Close:0.000031%
  Sat, 25 Mar 2023 16:26:52 INFO [Validate] Val_Acc:71.618622%  |  Open_ACC:57.222225%   |  Close_ACC:81.180809%
  Sat, 25 Mar 2023 16:26:53 INFO [Result] The best acc is 71.618622% at epoch 62
  ```
- RN50x4
  - 先進入 lib/utils/run.sh，然後執行這行做 resize:`python ./create_resized_images.py ../../data/data_rad/imgid2idx.json $IMAGEPATH 288 ../../data/data_rad/images288x288.pkl 3`
  - 進入 `configs/qcr_pubmedclipRN50x4_rad_16batchsize_withtfidf_nondeterministic.yaml`，將 PubMedClip 的 checkpoint path 填入 `CLIP_PATH`。
  - 執行 `python main.py --cfg configs/qcr_pubmedclipRN50x4_rad_16batchsize_withtfidf_nondeterministic.yaml`
  - 如果遇到 `new(): invalid datatype` 請修改 `dataset_RAD.py`（當初是在 inference trainset 時發現對`dataset_RAD.py`改動會有的錯誤）
  - 最高分數，沒錯，就是和 RN50 一樣的 71.618622%。
  ```
  Sat, 13 May 2023 00:30:28 INFO -------[Epoch]:61-------
  Sat, 13 May 2023 00:30:28 INFO [Train] Loss:0.000502 , Train_Acc:99.151436%
  Sat, 13 May 2023 00:30:28 INFO [Train] Loss_Open:0.000032 , Loss_Close:0.000031%
  Sat, 13 May 2023 00:30:32 INFO [Validate] Val_Acc:71.618622%  |  Open_ACC:58.563538%   |  Close_ACC:80.370369%
  Sat, 13 May 2023 00:30:34 INFO [Result] The best acc is 71.618622% at epoch 61
  ```
- ViT
  ```
  Sun, 14 May 2023 04:09:36 INFO -------[Epoch]:178-------
  Sun, 14 May 2023 04:09:36 INFO [Train] Loss:0.001017 , Train_Acc:99.053528%
  Sun, 14 May 2023 04:09:36 INFO [Train] Loss_Open:0.000017 , Loss_Close:0.000095%
  Sun, 14 May 2023 04:09:37 INFO [Validate] Val_Acc:71.175163%  |  Open_ACC:59.444447%   |  Close_ACC:78.966789%
  Sun, 14 May 2023 04:09:39 INFO [Result] The best acc is 71.175163% at epoch 178
  ```
## ⛔️ 1. Try fine-tuning with `roco`:

Can't find the `/train/radiologytraindata.csv` in ROCO dataset repo.


## ✅ 2. Try training the data on `VQA-RAD`


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
- 我不確定哪裡可以下載 `imgid2idx.json` 的檔案。已經修好（`run.sh` 內忘記執行）。
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
- 使用 `lib/utils/run.sh` 後產出的中間產物，會導致type_classifier checkpoint loading 錯誤。
[issue with VQA-RAD training](https://github.com/sarahESL/PubMedCLIP/issues/9)
因此根據作者建議將 Awenbocc/med-vqa 的 `data` 資料夾複製到 `QCR_PubMedCLIP/data/data_rad` 下。

- `clip` 的下載要按照：[openai/clip](https://github.com/openai/CLIP#usage)。
- 留意 cuda driver 版本問題。
- `QCR_PubMedCLIP/lib/utils` 資料夾下的 `run.sh` 裡面只要執行 create_resized_images 的 scripts 就好，其他的檔案用 `Awenbocc/med-vqa` 的。
- 可以用這個指令確認 $PWD 底下的.jpg images 數量：`find . -name "*.jpg" | wc -l`

## ✅ 2.1 Training log & Tensorboards (2023/03/25)

Dataset for VQA-Rad, see [Awenbocc/med-vqa](https://github.com/Awenbocc/med-vqa) (by checking issue QQ).
```
Sat, 25 Mar 2023 16:26:49 INFO -------[Epoch]:62-------
Sat, 25 Mar 2023 16:26:49 INFO [Train] Loss:0.000482 , Train_Acc:99.053528%
Sat, 25 Mar 2023 16:26:49 INFO [Train] Loss_Open:0.000029 , Loss_Close:0.000031%
Sat, 25 Mar 2023 16:26:52 INFO [Validate] Val_Acc:71.618622%  |  Open_ACC:57.222225%   |  Close_ACC:81.180809%
Sat, 25 Mar 2023 16:26:53 INFO [Result] The best acc is 71.618622% at epoch 62
```
Train Acc 大概在 25 epochs 時就幾乎達到最高峰。
Closed 的 VAcc 很不穩定，有個 2 ~ 3 % 之間的劇烈下跌。
Open 的 VAcc 整體呈現上升趨勢。
整體而言和 Paper 的數據有約莫 0.5% 的差距（較低）。

![](QCR_PubMedCLIP/output/qcr/pubmedclipRN50_ae/roco/VQARAD/QCR.CLIPRN50.AE.ROCO.VQARAD.16batchsize.200epoch.withTFIDF.nondeterministic/imgs/2023_0405_訓練結果tensorboard.png)

## ✅ 2.2 Saving Pretrained checkpoints
saved at `/home/nanaeilish/projects/mis/PubMedCLIP/QCR_PubMedCLIP/output/qcr/pubmedclipRN50_ae`
## ✅ 3. Exporting the prediction on `VQA-RAD`

testing 指令：
`python main.py --test True --cfg configs/qcr_pubmedclipRN50_ae_rad_16batchsize_withtfidf_nondeterministic.yaml`

Add these (or at least your configured paths) into the config file `--cfg`.

```
MODEL_FILE: "PubMedCLIP/QCR_PubMedCLIP/output/qcr/pubmedclipRN50_ae/roco/VQARAD/QCR.CLIPRN50.AE.ROCO.VQARAD.16batchsize.200epoch.withTFIDF.nondeterministic/62_best.pth"
  RESULT_DIR: "PubMedCLIP/QCR_PubMedCLIP/output/qcr/pubmedclipRN50_ae/roco/VQARAD/QCR.CLIPRN50.AE.ROCO.VQARAD.16batchsize.200epoch.withTFIDF.nondeterministic/results"
```
### Issue [solved]
在寫成 testfile 時他的 Predicted_answer 都只寫出 closed_logits
是 tensor 的形式，沒有轉換成文字，要自己 decode。
[Hackmd Notes: How to get the inference data](https://hackmd.io/@NanaEilish727/pmclip)

### 成績
`PubMedCLIP/QCR_PubMedCLIP/output/qcr/pubmedclipRN50_ae/roco/VQARAD/QCR.CLIPRN50.AE.ROCO.VQARAD.16batchsize.200epoch.withTFIDF.nondeterministic/medVQA.log` 中的 validation best acc is `71.618622%` at epoch 62，最後 test 使用 epoch 62 的checkpoint的成績是 `71.175163%`。根據 `/QCR_PubMedCLIP/main.py`，test mode 下依然使用 val_loader，因此不確定為何會有分數差。
```
451 179.0 272.0   # 全部，Open，Close 的數量
[Validate] Val_Acc:71.175163%  |  Open_ACC:56.424580%   |  Close_ACC:80.882355%
```

## ✅ 4. Use the prediction and the data itself for data EDA
[Visualization](./extra/0.1_vis.ipynb)
1. Data Example。
2. `question_type` 的數量。
3. `question_type` level 的 EDA（類別正確與錯誤率）。
4. 顯示答錯例子。
5. 梳理 method 流程。


## Todo:
0. 完整的設定檔理解（data preproc 到底做到哪，有 resize 嗎，有旋轉嗎）
1. xai 套件如 LIME
2. medical preprocessing （傅立葉轉換、輪廓清晰 filter）工具包試用

