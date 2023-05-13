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
WW (window width) = (CT下限值 - CT上限值)
WL (window level) = (CT下限值 + CT上限值) / 2

HEAD CT 判讀： 通常有腦白質、灰質、脊髓液、凝固的血塊、骨頭 http://www.tma.org.tw/ftproot/2022/20220217_14_11_54.pdf
ABD CT 判讀： 通常有胃、肝、脾臟、小腸、大腸、膀胱、骨盆 http://www.shensc.tw/2018/11/ct.html
"""
_DEFAULT_HU_TRANSFORM_PARAMS = {
    'HEAD': (16, 72),
    'ABD': (45, 48)
}

_MORPHOLOGY_KERNEL = np.ones((5, 5), np.uint8)
_GAUSS_VALUE = 5
_CANNY_MIN = 230
_CANNY_MAX = 250