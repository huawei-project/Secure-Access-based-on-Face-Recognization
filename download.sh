#! bin/bash
###
 # @Description: 
 # @Version: 1.0.0
 # @Author: louishsu
 # @E-mail: is.louishsu@foxmail.com
 # @Date: 2020-01-05 17:12:46
 # @LastEditTime : 2020-01-05 17:15:57
 # @Update: 
 ###

wget -O ckpt/PNet.pkl https://github.com/isLouisHsu/MTCNN_Darknet/releases/download/v1.0/PNet.pkl
wget -O ckpt/RNet.pkl https://github.com/isLouisHsu/MTCNN_Darknet/releases/download/v1.0/RNet.pkl
wget -O ckpt/ONet.pkl https://github.com/isLouisHsu/MTCNN_Darknet/releases/download/v1.0/ONet.pkl
wget -O ckpt/MobileFacenet.pkl https://github.com/isLouisHsu/MobileFaceNet_ArcFace_Darknet/blob/master/pretrained/MobileFacenet_best.pkl