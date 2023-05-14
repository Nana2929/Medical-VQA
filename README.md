# Preprocessing
主要流程為透過指定要處理的 modality 並讀入所有相關圖片後，程式會執行已經定義好的 pipeline，並把處理後的影像存到指定的目錄中。其中每個處理階段會分開存，因此最後一個階段的結果即是所有流程跑過的結果。
## 如何使用？
可以直接執行 [main.py](https://github.com/Nana2929/Medical-VQA/blob/preprocess/main.py)，
```
python main.py
```
並有幾個參數可以調整：

| Parameter     |      Default       | Description                               |                    Constraints           |
|:--------------|:------------------:|:------------------------------------------|:------------------------------------------|
| -t --target   | HEAD_CT | target modaility | HEAD_CT / HEAD_MRI / CHEST / ABD
| -o --output   | ./outputs/ | output folder name |
| --trainset |         trainset.json          | file name of training set | only .json file
| --testset |         testset.json         | file name of testing set | only .json file
<br />

## 加新的 Preprocess 方法
1. 在 [preprocessing](https://github.com/Nana2929/Medical-VQA/tree/preprocess/preprocessing) 中加入Function，可以加在已經存在的 module，或是沒有符合條件的則可以新增 module。
2. 在 [\_\_init\_\_.py](https://github.com/Nana2929/Medical-VQA/blob/preprocess/preprocessing/__init__.py) import 加入新增的方法，並找到 _DEFAULT_PIPELINE_STEPS，並根據要處理的 modality 把新增的 function 加進去 pipeline 中。格式可以參考以下面 Tuple 格式：(\<function_name\>, \<arguments...\>)。需注意所有的fuction第一個參數必為要處理的影像，因此不用放在 tuple 中，也必須回傳處理後的圖片，以供其他方法能接續使用。以下給幾個範例。

    (1) 參數只有影像的情況：
    ```
    def test_a(img):
        # do something
        result = do_something(img)
        return result

    _DEFAULT_PIPELINE_STEPS = {
        'HEAD_CT': [
            (test_a, ) # note the comma 
        ],
        ...
    }
    ```
    (2) 除了影像外，有其餘參數：
    ```
    def test_b(img, a, b):
        # do something
        result = do_something(img)
        return result

    _DEFAULT_PIPELINE_STEPS = {
        'HEAD_CT': [
            (test_b, 1, 'test')
        ],
        ...
    }
    ```
