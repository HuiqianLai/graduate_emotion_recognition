# graduate_emotion_recognition
use cnn,nn and svm to recognize people's face

I use three data sets：Fer2013, CK+(The Extended Cohn-Kanade Dataset) and JAFFE(The Japanese Female Facial Expression)
Download link:
1/
Fer2013 *from Kaggle: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
Fer2013 *from Baiduyun: 
链接：https://pan.baidu.com/s/1zcGDgzW8wfhHJTGZRISM1Q 
提取码：2m6a 
复制这段内容后打开百度网盘手机App，操作更方便哦
2/
CK+ *from CMU: http://www.consortium.ri.cmu.edu/ckagree/
CK+ *from Baiduyun: 
链接：https://pan.baidu.com/s/1TgYUflagmCTwcmjIuowFew 
提取码：yb7r 
复制这段内容后打开百度网盘手机App，操作更方便哦
3/
JAFFE: http://www.kasrl.org/jaffe_info.html

Steps:
1/Create new opt and input folders under the path
and under the opt folder,new: jaffe-cnn,jaffe-nn,jaffe-svm,fer-cnn,fer-nn,fer-svm,ck-cnn,ck-nn,ck-svm folder
2/Put all the code and data set in the input folder
CK_data.h5 is CK+ dataset; 
data.h5 is fer2013 dataset;
jaffe_mean_data.mat is JAFFE dataset;
3/Use Jupyter Notebook or cmd to run my code
