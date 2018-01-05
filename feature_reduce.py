# -*- coding: utf-8 -*-
'''
基于LDA，对人脸数据集进行维数约减，并计算分类准确率
'''
from __future__ import print_function
import matplotlib.pyplot as plt
from time import time
import logging  

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

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

h1=192
w1=168
h2=112
w2=92

target_names1 = []
target_names2 = []  
for i in range(1,11):  
    names = "person" + str(i)  
    target_names1.append(names)

for i in range(1,41):  
    names = "person" + str(i)  
    target_names2.append(names)

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
X_train1,X_test1,y_train1,y_test1=train_test_split(data_1,label_1,test_size=130,random_state=0,stratify=label_1)
X_train2,X_test2,y_train2,y_test2=train_test_split(data_2,label_2,test_size=80,random_state=0,stratify=label_2)

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
clf1=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
clf2=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
scores1=cross_val_score(clf1, X_train1_new, y_train1, cv=5)
scores2=cross_val_score(clf2, X_train2_new, y_train2, cv=5)

#print("5-fold cross validation scores:")
print("5-fold cross validation accuracy on data1: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))
print("5-fold cross validation accuracy on data2: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))


#使用训练好的分类器对测试数据进行预测
y_test1_pred=clf1.fit(X_train1_new, y_train1).predict(X_test1_new)
y_test2_pred=clf2.fit(X_train2_new, y_train2).predict(X_test2_new)

#计算测试数据集上的准确率
comp1=y_test1_pred-y_test1#比较预测的结果和真实类标
comp2=y_test2_pred-y_test2
ac_rate1=np.sum(comp1==0)/X_test1_new.shape[0]#获得正确率
ac_rate2=np.sum(comp2==0)/X_test2_new.shape[0]

print("Accuracy on test data on data1: %0.4f" %ac_rate1)
print("Accuracy on test data on data2: %0.4f" %ac_rate2)

def plot_gallery(images, titles, h, w, n_row, n_col):  
    """Helper function to plot a gallery of portraits"""  
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))  
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)  
    for i in range(n_row * n_col):  
        plt.subplot(n_row, n_col, i + 1)  
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)  
        plt.title(titles[i], size=12)  
        plt.xticks(())  
        plt.yticks(())  


def title(y_pred, y_test, target_names, i): 
	#展示测试集预测结果图的题目 
    pred_name = target_names[y_pred[i]-1]  
    true_name = target_names[y_test[i]-1]  
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)  

prediction_titles1 = [title(y_test1_pred, y_test1, target_names1, i)  
                     for i in range(y_test1_pred.shape[0])] 

prediction_titles2 = [title(y_test2_pred, y_test2, target_names2, i)  
                     for i in range(y_test2_pred.shape[0])]  
                    
'''
plot the result of the prediction on a portion of the test set 1  
'''
plot_gallery(X_test1, prediction_titles1, h1, w1, 10, 13)
'''
plot the result of the prediction on a portion of the test set 2  
'''
plot_gallery(X_test2, prediction_titles2, h2, w2, 10, 8)