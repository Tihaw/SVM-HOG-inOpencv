# SVM-HOG-inOpencv
a simple interface using svm to detect object by HOG descriptors, using opencv libs.该程序使用SVM通过HOG特征进行图像目标的检测。

because i try to detect very small object and its size varies, so multiple windows are set, the speed is not good.
non-maxinum depress is considered.
由于我是用来检测小物体的，HOG的参数需要按需调整，该项目速度不快，程序中引入了非极大值抑制。

training pics and test pics are not included.没有上传训练库和测试库。

训练库和测试目录是：
you can put them in:
 positiveDir = "DMset-rectified/train/positive";
 negativeDir ="DMset-rectified/train/negative";
 testFile = "DMset-rectified/test";
