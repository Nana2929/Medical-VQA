import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import albumentations as A
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import joblib
# 定義資料增強的方法
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
])
# 定義讀取圖像並計算紋理特徵的函數
def extract_texture_features(image):
    # 將彩色影像轉為灰度影像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 設定紋理分析的參數
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    levels = 256
    symmetric = True
    normed = True
    # 計算灰度共生矩陣（GLCM）
    glcm = graycomatrix(gray, distances=distances, angles=angles,
                        levels=levels, symmetric=symmetric, normed=normed)
    # 計算GLCM的六個紋理特徵
    contrast = graycoprops(glcm, 'contrast') # 對比度 亮度的对比情况
    dissimilarity = graycoprops(glcm, 'dissimilarity') # 不相似度
    homogeneity = graycoprops(glcm, 'homogeneity') # 同質性 测量图像的局部均匀性
    energy = graycoprops(glcm, 'energy') # 能量 图像纹理的灰度变化稳定程度的度量
    ASM = graycoprops(glcm, 'ASM') # 灰度共生矩陣的總和 ASM有较大值，若G中的值分布较均匀（如噪声严重的图像），则ASM有较小的值。
    correlation = graycoprops(glcm, 'correlation') # 自相关 图像纹理的一致性



    # 將六個特徵合併為一個特徵向量
    features = np.hstack([contrast, dissimilarity, homogeneity, energy,ASM,correlation])
    return features

# 讀取資料夾中的影像，並計算紋理特徵
directory = 'data/images/'
data = pd.read_csv('./output2.csv')
label_name = ['X-Ray', 'CT', 'MRI']
image_paths = []
labels = []

for i in range(len(data)):
    image_paths.append(data['imageID'][i])
    labels.append(label_name.index(data['class'][i]))

# 提取影像特徵
features = []
for path in image_paths:
    image = cv2.imread(directory+path)
    image = cv2.resize(image, (512, 512))
    # # 做資料增強
    # transformed = transform(image=image)
    # image_augmented = transformed['image']
    # feature = extract_texture_features(image_augmented)
    # features.append(feature)
    feature = extract_texture_features(image)
    features.append(feature)
features = np.array(features)

# 切分資料集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

X_train2D = X_train.reshape(X_train.shape[0], -1)
X_test2D = X_test.reshape(X_test.shape[0], -1)


# 訓練機器學習模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train2D, y_train)
joblib.dump(model, 'LR_model')

# 預測測試集
y_pred = model.predict(X_test2D)


# 準確率
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# 精確率
precision = precision_score(y_test, y_pred, average='weighted')
print('Precision:', precision)

# 召回率
recall = recall_score(y_test, y_pred, average='weighted')
print('Recall:', recall)

# F1 Score
f1 = f1_score(y_test, y_pred, average='weighted')
print('F1 Score:', f1)


cm = confusion_matrix(y_test, y_pred)
# Create a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=label_name, yticklabels=label_name)

# Add labels and title to the plot
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

