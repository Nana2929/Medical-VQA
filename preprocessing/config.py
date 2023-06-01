import numpy as np
"""
常見人體組織CT值(HU):
組織 CT值
------------
骨組織 >400
肝臟 50 ~ 70
鈣值 80 ~ 300
脾臟 35 ~ 60
血塊 64 ~ 84
胰腺 30 ~ 55
腦白質 25 ~ 34
腎臟 25 ~ 50
腦灰質 28 ~ 44
肌肉 40 ~ 55
腦脊液 3 ~ 8
膽襄 10 ~ 30
血液 13 ~ 32
甲狀腺 50 ~ 90
血漿 3 ~ 14
脂肪 -20 ~ -100
滲出液 >15
水 0

計算方式：
WL: window level
WW: window width
CT下限值 = WL - (WW / 2)
CT上限值 = WL + (WW / 2)

HEAD CT 判讀： 通常有腦白質、灰質、脊髓液、凝固的血塊、骨頭 http://www.tma.org.tw/ftproot/2022/20220217_14_11_54.pdf
ABD CT 判讀： 通常有胃、肝、脾臟、小腸、大腸、膀胱、骨盆 http://www.shensc.tw/2018/11/ct.html
"""
_DEFAULT_HU_TRANSFORM_PARAMS = {'HEAD': (48, 68), 'ABD': (70, 104)}
# define the low-quality images
# see preprocess branch: image-inspection/measure_ct_noise.py
# Q4 images
_CT_NOISE_FUNCS = [
    'median_filter',
    'wiener_filter',
]
_TO_APPLY_ABD_FILTERS = [
    'synpic24878.jpg', 'synpic43648.jpg', 'synpic19605.jpg', 'synpic34515.jpg',
    'synpic33689.jpg', 'synpic22310.jpg', 'synpic47191.jpg', 'synpic50949.jpg',
    'synpic42290.jpg', 'synpic30324.jpg', 'synpic22286.jpg', 'synpic42182.jpg',
    'synpic23571.jpg', 'synpic29771.jpg', 'synpic23008.jpg', 'synpic28210.jpg',
    'synpic54823.jpg', 'synpic52828.jpg', 'synpic38630.jpg', 'synpic31232.jpg',
    'synpic34054.jpg', 'synpic22156.jpg', 'synpic43433.jpg', 'synpic45115.jpg',
    'synpic28569.jpg', 'synpic24967.jpg'
]
_SKIP_TILT_CORRECTION = [
    # HEAD MRI
    'synpic26925.jpg',
    'synpic37605.jpg',
    'synpic40096.jpg',
    'synpic47356.jpg',
    'synpic46764.jpg',
    'synpic47191.jpg',
    'synpic50848.jpg',
    'synpic54004.jpg',
    'synpic53287.jpg',
    'synpic59935.jpg',
    'synpic20626.jpg',
    'synpic31928.jpg',
    'synpic34836.jpg',
    'synpic38858.jpg',
    'synpic46720.jpg',
    'synpic49381.jpg',
    'synpic56422.jpg',
    'synpic34854.jpg',
    'synpic34947.jpg',
    'synpic39460.jpg',
    'synpic44995.jpg',
    'synpic51709.jpg',
    'synpic55583.jpg',
    'synpic56061.jpg',
    'synpic57935.jpg',
    # HEAD CT
    'synpic57813.jpg',
    'synpic23631.jpg',
    'synpic40500.jpg'
]

_MORPHOLOGY_KERNEL = np.ones((5, 5), np.uint8)
_GAUSS_VALUE = 5
_CANNY_MIN = 230
_CANNY_MAX = 250
_FCM_NORM_VALUE = 0.8