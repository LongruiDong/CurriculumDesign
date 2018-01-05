# -*- coding: utf-8 -*-
'''
基于LDA，对人脸数据集进行维数约减，并计算分类准确率
'''
from __future__ import print_function
from time import time
import logging  

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

PICTURE_PATH1 = "D:\\mycollege\\term7\\keshe\\CroppedYale"#yaleb人脸数据  10*65
PICTURE_PATH2 = "D:\\mycollege\\term7\\keshe\\att_faces"#at&t人脸数据     40*10
#载入npy文件，得到整个数据集的ndarray
data_1=np.load(PICTURE_PATH1 + "\\"+"alldataset.npy")
label_1=np.load(PICTURE_PATH1 + "\\"+"alldatalabel.npy")
data_2=np.load(PICTURE_PATH2 + "\\"+"alldataset.npy")
label_2=np.load(PICTURE_PATH2 + "\\"+"alldatalabel.npy")

n_samples1,n_features1=data_1.shape
n_classes1 = len(np.unique(label_1))
n_samples2,n_features2=data_2.shape
n_classes2 = len(np.unique(label_2))
#输出数据集信息
print("Total 1th dataset size:")
print("n_samples: %d" % n_samples1)
print("n_features: %d" % n_features1)
print("n_classes: %d" % n_classes1)
print("Total 2nd dataset size:")
print("n_samples: %d" % n_samples2)
print("n_features: %d" % n_features2)
print("n_classes: %d" % n_classes2)
#将数据集划分为训练集和测试集
X_train1,X_test1,y_train1,y_test1=train_test_split(data_1,label_1,test_size=50,random_state=0,stratify=label_1)
X_train2,X_test2,y_train2,y_test2=train_test_split(data_2,label_2,test_size=40,random_state=0,stratify=label_2)

#对训练数据和测试数据降维
lda_1=LinearDiscriminantAnalysis(solver='svd')
lda_1=lda_1.fit(X_train1,y_train1)
X_train1_new=lda_1.transform(X_train1)
X_test1_new=lda_1.transform(X_test1)

lda_2=LinearDiscriminantAnalysis(solver='svd')
lda_2=lda_2.fit(X_train2,y_train2)
X_train2_new=lda_2.transform(X_train2)
X_test2_new=lda_2.transform(X_test2)
#输出降维后的数据集信息
print("Total 1th transformed dataset size:")
print("n_samples_train: %d" % X_train1_new.shape[0])
print("n_features_train: %d" % X_train1_new.shape[1])
print("n_samples_test: %d" % X_test1_new.shape[0])
print("n_features_test: %d" % X_test1_new.shape[1])
print("Total 2nd transformed dataset size:")
print("n_samples_train: %d" % X_train2_new.shape[0])
print("n_features_train: %d" % X_train2_new.shape[1])
print("n_samples_test: %d" % X_test2_new.shape[0])
print("n_features_test: %d" % X_test2_new.shape[1])
#使用lda进行分类验证
#使用5折交叉验证来展示分类效果













#使用训练好的分类器对测试数据进行预测


