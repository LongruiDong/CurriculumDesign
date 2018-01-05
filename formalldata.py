# -*- coding: utf-8 -*-
'''
读取数据集pgm文件脚本。
转化为ndarray形式保存为.npy文件，每一行都是一个图片
'''
import cv2#opencv库
import numpy as np

PICTURE_PATH1 = "D:\\mycollege\\term7\\keshe\\CroppedYale"#yaleb人脸数据  
PICTURE_PATH2 = "D:\\mycollege\\term7\\keshe\\att_faces"#at&t人脸数据
all_data_set = []  
all_data_label = []    

def get_Image(): 
#读取pgm图像并转为一列，返回图片的长宽 
    for i in range(1,41): #11 类数
        for j in range(1,11):#66  每一类的样本数
            path = PICTURE_PATH2 + "\\" + str(i) + "\\"+ str(j) + ".pgm"  
            img = cv2.imread(path,cv2.IMREAD_UNCHANGED) #读取pgm图片   
            h,w = img.shape  
            img_col = img.reshape(h*w)  
            #把图像和类标添加到总数据集
            all_data_set.append(img_col)  
            all_data_label.append(i)  
    return h,w  

def main():
	h,w = get_Image()  
	#转为ndarray      
	X = np.array(all_data_set)    
	y = np.array(all_data_label)
	np.save(PICTURE_PATH2 + "\\"+"alldataset.npy",X)
	np.save(PICTURE_PATH2 + "\\"+"alldatalabel.npy",y) 
	n_samples,n_features = X.shape  
	n_classes = len(unique(y))  
	target_names = []  
	for i in range(1,11):  
		names = "person" + str(i)  
		target_names.append(names)  
    #输出数据集的大小  
	print("Total dataset size:")  
	print("n_samples: %d" % n_samples)  
	print("n_features: %d" % n_features)  
	print("n_classes: %d" % n_classes)  

main()