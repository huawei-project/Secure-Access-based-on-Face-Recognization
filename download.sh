#! bin/bash
###
 # @Description: 
 # @Version: 1.0.0
 # @Author: louishsu
 # @E-mail: is.louishsu@foxmail.com
 # @Date: 2020-01-05 17:12:46
 # @LastEditTime : 2020-01-06 17:39:05
 # @Update: 
 ###

wget -O ckpt/PNet.pkl https://github.com/isLouisHsu/MTCNN_Darknet/releases/download/v1.0/PNet.pkl
wget -O ckpt/RNet.pkl https://github.com/isLouisHsu/MTCNN_Darknet/releases/download/v1.0/RNet.pkl
wget -O ckpt/ONet.pkl https://github.com/isLouisHsu/MTCNN_Darknet/releases/download/v1.0/ONet.pkl

wget -O weights/PNet.weights https://github.com/isLouisHsu/MTCNN_Darknet/releases/download/v1.0/PNet.weights
wget -O weights/RNet.weights https://github.com/isLouisHsu/MTCNN_Darknet/releases/download/v1.0/RNet.weights
wget -O weights/ONet.weights https://github.com/isLouisHsu/MTCNN_Darknet/releases/download/v1.0/ONet.weights

wget -O ckpt/MobileFacenet.pkl        https://github.com/isLouisHsu/MobileFaceNet_ArcFace_Darknet/releases/download/v1.0/MobileFacenet_best.pkl
wget -O weights/mobilefacenet.weights https://github.com/isLouisHsu/MobileFaceNet_ArcFace_Darknet/releases/download/v1.0/mobilefacenet.weights
