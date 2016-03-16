//revised in 2016-3-16.
#include "FarDimObj.h"
extern "C"
{
#include "graph.h"
};

struct rectBRConfience
{
	Rect dmtx;
	float svmPreConfidence;
	bool ifMax;
};

enum Command{Train,Predict};
Command inputOrders = Predict;
string positiveDir = "DMset-rectified/train/positive";
string negativeDir ="DMset-rectified/train/negative";
string testFile = "DMset-rectified/test";

//图像路径和类别
vector<string> img_path;
vector<int> img_catg;  
//扫描窗口 最小20*20
const int minWinWidth = 20;
const int minWinHeight = 20;
//扫描窗口 最大60*60
const int maxWinWidth = 60;
const int maxWinHeight = 60;

//储存数据data、响应response矩阵
Mat data_mat, res_mat; 
//程序计时
DWORD time1,time2; 
//HOG描述符，定义 注意有关参数和图像大小关系
// CV_WRAP HOGDescriptor() : winSize(64,128), blockSize(16,16), blockStride(8,8),
// 	cellSize(8,8), nbins(9), derivAperture(1), winSigma(-1),
// 	histogramNormType(HOGDescriptor::L2Hys), L2HysThreshold(0.2), gammaCorrection(true),
// 	nlevels(HOGDescriptor::DEFAULT_NLEVELS)
//(winSize - blockSize) %blockStride ==0  && blockSize % cellSize == 0
HOGDescriptor *hog=new HOGDescriptor(cvSize(trainImgWidth,trainImgHeight),
	cvSize(20,20),cvSize(10,10),cvSize(10,10), 9,1,-1,0,0.2, true,64);

void trainDataSet();
void testData();
void getFiles( string path, string exd, vector<string>& files );
void getExemplarAndSetCategory( string posPath, int posCateg, string negPath, int negCateg);
void getExemplarFeatures();
void testImg();
void fExtractByHOG(Mat &img, vector<float> &descriptors);
int Rect1OverlapsRect2(Rect &rect1, Rect &rect2);
void nms_pickDmtxRects(vector<vector<rectBRConfience>> &dmtxRects);
Rect Rect1PlusRect2(Rect &rect1, Rect &rect2);

int main(int argc, char** argv)    
{   
	char cmmd = 0;
	cout <<endl <<endl <<"    welcome to the SVM+Feature exp sys........."<<endl <<endl;
	cout << "Pls input command and press 'enter'..." <<endl;
	cout << "t or T for Training, while p or P for Predict" <<endl;
	while(1)
	{
		cin >> cmmd;
		if (cmmd == 't' || cmmd == 'T')
		{
			inputOrders = Train;
			cout << "Program going to do SVM Train..." <<endl;
			waitKey(1000);
			break;
		}
		else if (cmmd == 'p' || cmmd == 'P')
		{
			inputOrders = Predict;
			cout << "Program going to do SVM Predict..." <<endl;
			waitKey(1000);
			break;
		}
	}

	if (inputOrders == Train)
	{
		getExemplarAndSetCategory(positiveDir, 1, negativeDir, 0);
		getExemplarFeatures();
		trainDataSet();
	}
	else
	{
		testImg();
	}

	system("pause");
	return 0;  
}  

//训练SVM。
void trainDataSet()
{
	CvSVM svm;
	CvSVMParams param;  
	CvTermCriteria criteria;    
	criteria = cvTermCriteria( CV_TERMCRIT_ITER, 1000, FLT_EPSILON );   
	/*	CvSVMParams( int svm_type, int kernel_type,
	double degree, double gamma, double coef0,
	double Cvalue, double nu, double p,
	CvMat* class_weights, CvTermCriteria term_crit );*/
	param = CvSVMParams( CvSVM::C_SVC, CvSVM::LINEAR, 
		10.0, 0.09, 1.0, 
		0.01, 0.5, 1.0,
		NULL, criteria );

	cout <<endl <<endl <<"    start training ,please wait........."<<endl <<endl;
	time1 = timeGetTime();
	svm.train_auto( data_mat, res_mat, Mat(), Mat(), param );   
	//svm.train_auto(data_mat,res_mat,Mat(), Mat(), param);
	svm.save( "SVM_DATA.xml" ); 

	cout<<"training ends,"<<"which took "<<(timeGetTime()-time1)/1000<<" s"<<endl;
}

//读取文件目录和设置01类别
void getExemplarAndSetCategory( string posPath, int posCateg, string negPath, int negCateg)
{
	int curTotalFiles = 0;

	cout <<endl <<endl <<"    begin find data set files..."<<endl <<endl;

	//设置positive文件和类别
	getFiles(posPath, "*", img_path);
	cout <<"found "  << img_path.size() << " positive files" <<endl;
	//记录总样本数目、现在是positive数目
	curTotalFiles = img_path.size();

	//设置positive文件夹下 类别为1.
	for ( int i = 0; i < static_cast<int>(img_path.size()); i++)
	{
		img_catg.push_back(posCateg);
	}

	//同样设置negative
	getFiles(negPath, "*", img_path);
	cout <<"found "  << img_path.size() - curTotalFiles << " negative files" <<endl;
	//设置negative文件夹下 类别为0.注意从后面添加
	for ( int i = curTotalFiles; i < static_cast<int>(img_path.size()); i++)
	{
		img_catg.push_back(negCateg);
	}

	//记录总样本数目、现在是positive+negative数目
	curTotalFiles = img_path.size();

	cout <<"totally "  << curTotalFiles << " files";
}


void fExtractByHOG( Mat &img, vector<float> &descriptors )
{
	hog->compute(img, descriptors, Size(1,1), Size(0,0)); 
}

//记录在data_mat
void getExemplarFeatures()
{ 
	int nImgNum = img_path.size();
	//读入响应矩阵01010101, not copy data
	res_mat = Mat( img_catg, false); 
	Mat src;
	//训练集合的图像规定图像大小50*50。用宏定义还是变量?
	Mat trainImg = Mat::zeros(trainImgHeight, trainImgWidth, CV_8UC1);//需要分析的图片  

	cout <<endl <<endl <<"    Start reading & processing IMG..."<<endl <<endl;
	time1 = timeGetTime();
	for( int i = 0; i < static_cast<int>(img_path.size()); i++ )
	{  
		//opencv二值化的方式CV_LOAD_IMAGE_GRAYSCALE CV_BGR2GRAY和 直接读入 区别?
		src = imread(img_path[i], CV_LOAD_IMAGE_GRAYSCALE);   
		trainImg = src;

		vector<float>descriptors;

		//计算特征
		fExtractByHOG(trainImg, descriptors);

		if (i==0)
		{
			data_mat = Mat::zeros( nImgNum, descriptors.size(), CV_32FC1 ); 
		}

		Mat tmp(descriptors),tmpT;
		tmpT = tmp.t();
		tmpT.copyTo(data_mat.row(i));
		if (i%10 == 0)
		{
			cout << endl << i / 10;
		}
		cout << ".";
	}  // for each train pic

	cout << endl;
	res_mat = Mat(img_catg, true);

	cout <<"finished extracting feature vector, HOG dims: "<< data_mat.cols;
	cout <<". Time consume:"<<(timeGetTime()-time1)/1000<<" s";
}

void testImg()
{
	//test
	CvSVM svm;

	cout <<endl <<endl <<"    start reading SVM_DATA.xml ,please wait........."<<endl <<endl;
	time1 = timeGetTime();
	svm.load("SVM_DATA.xml");
	cout<<"loading time consume"<<timeGetTime()-time1<<"ms"<<endl;

	vector<string> img_tst_path;

	vector<vector<Point>> con;

	//Graph positiveWin;
	//vector<rectBRConfience> dmtxRects;
	vector<vector<rectBRConfience>> dmtxRects;
	rectBRConfience tmpRectConfidence;
	Rect tmpBR;
	int rectLength = 0;
	int testWinCount = 0;
	int positivePreWinCount = 0;

	//test for show, testGray for detect, GrayCopy for canny, ImgRegion for "testGray roi"
	Mat test, testGray, testGrayCopy, testImgRegion;
	/*	char curFilePath[512];*/
	ofstream predict_txt( "SVM_PREDICT.txt" );

	getFiles(testFile,"tiff", img_tst_path);

	//for each input file , select some of the contours and SVMpredict it
	for (int i = 0;i < static_cast<int>(img_tst_path.size());i++)
	{
		//init params
		time1 = timeGetTime();
		testWinCount = 0;
		positivePreWinCount = 0;
		rectLength = 0;
		con.clear();
		dmtxRects.clear();

		cout << "current file: "<< img_tst_path[i].c_str() <<"..."<<endl;
		//read and preprocess
		test = imread( img_tst_path[i], 1);
		cvtColor(test, testGray, CV_BGR2GRAY);
		testGray.copyTo(testGrayCopy);
		Canny(testGrayCopy, testGrayCopy, 20, 80, 3);

		// 		imshow("asdf",testGray);
		// 		waitKey(0);

		findContours(testGrayCopy, con, CV_RETR_LIST,  CV_CHAIN_APPROX_NONE);

		//process each contour
		for (int i =0; i < static_cast<int>(con.size());i++)
		{
			if (con[i].size()>40 && con[i].size() < 400)
			{
				//change this rect to a square
				tmpBR = boundingRect(con[i]);
				rectLength = max(tmpBR.width,tmpBR.height);

				//bigger than 60*60, do not process :P
				if (rectLength < 60)
				{
					tmpBR.width = rectLength;
					tmpBR.height = rectLength;

/*******************here begins the SVM predict*********************************/
					//this tmpBR will be a SVM predict input
					testWinCount++;

					//use yellow color to mark candidate rects
					rectangle(test, tmpBR,Scalar(0,255,255));

					//setup a img roi
					if ( tmpBR.y + tmpBR.height >= testGray.rows
						|| tmpBR.x + tmpBR.width >= testGray.cols)
					{
						continue;
					}
					testImgRegion = testGray(tmpBR);

					//cal feature
					vector<float>descriptors;
					if (tmpBR.height != trainImgHeight || tmpBR.width != trainImgWidth)
					{
						//又一个好参数来调试测试. 插值方式
						resize(testImgRegion, testImgRegion,
							cv::Size(trainImgWidth,trainImgHeight), 0, 0, CV_INTER_LINEAR);
					}

					fExtractByHOG(testImgRegion, descriptors);

					Mat tmp(descriptors),SVMtrainMat;
					SVMtrainMat = tmp.t();

					float svm_pre_distance = svm.predict(SVMtrainMat, true);
					int ret =  static_cast<int>( svm.predict(SVMtrainMat, false) ); 

					//pushback the positive class
					if (ret == 1)
					{
						tmpRectConfidence.dmtx = tmpBR;
						tmpRectConfidence.svmPreConfidence = svm_pre_distance;
						tmpRectConfidence.ifMax = false;

						bool ifCategoried = false;
						// push the first vector
						if (static_cast<int>(dmtxRects.size()) == 0)
						{
							dmtxRects.push_back(vector<rectBRConfience>());
							dmtxRects[0].push_back(tmpRectConfidence);
						}
						else
						{
							//check the class
							for (int j = 0;j<static_cast<int>(dmtxRects.size()) && !ifCategoried;j++)
							{
								for (int k = 0 ; k < static_cast<int>(dmtxRects[j].size());k++)
								{
									if (Rect1OverlapsRect2(tmpRectConfidence.dmtx, 
										dmtxRects[j][k].dmtx))
									{//i belongs to this class
										dmtxRects[j].push_back(tmpRectConfidence);
										ifCategoried = true; 
										break;
									}
								}//for (int k = 0 ; k < dmtxRects[j].size();k++)
							}//for each class of rect

							//cannot find a belonging category or class
							if (!ifCategoried)
							{// new a class
								dmtxRects.push_back(vector<rectBRConfience>());
								dmtxRects[dmtxRects.size() - 1].push_back(tmpRectConfidence);
							}

						}//switch the rects besides the 1st one.

					}//for each possible rects

					//count for test windows
					testWinCount ++;
				}
			}// if the contour size if appropriate
		}//for each contour

		//Non-maximum suppression
		nms_pickDmtxRects(dmtxRects);

		//show
		for (int i_rect = 0; i_rect< static_cast<int>(dmtxRects.size()); i_rect++ )
		{
			//mark positive green
			for (int j_rect = 0; j_rect< static_cast<int>(dmtxRects[i_rect].size()); j_rect++)
			{
				if (dmtxRects[i_rect][j_rect].ifMax == true)
				{
					positivePreWinCount++;
					rectangle(test, dmtxRects[i_rect][j_rect].dmtx,Scalar(0,255,0),2);
					break;
				}
			}
		}

		//report
		cout<<"*total time: "<< timeGetTime() - time1 <<"ms;"<<endl;
		cout<<"*total check windows: "<<testWinCount<<";"<<endl;
		cout<<"*positives:  "<<positivePreWinCount<<";"<<endl;

		resize(test,test, Size(test.cols/2, test.rows/2));
		//after all winSize detected.
		imshow("ANS", test);
		waitKey(0);

	}// for each file

	predict_txt.close();  
}

//0 means not overlapping. or return the area that overlaps
int Rect1OverlapsRect2(Rect &rect1, Rect &rect2)
{
	int minx = max(rect1.tl().x, rect2.tl().x);
	int miny = max(rect1.tl().y, rect2.tl().y);
	int maxx = min(rect1.br().x, rect2.br().x);
	int maxy = min(rect1.br().y, rect2.br().y);
	if (minx > maxx || miny > maxy)
	{
		return 0;
	}
	return (maxx-minx)*(maxy-miny);
}

//0 means not overlapping. or return the area that overlaps
Rect Rect1PlusRect2(Rect &rect1, Rect &rect2)
{
	int minx = max(rect1.tl().x, rect2.tl().x);
	int miny = max(rect1.tl().y, rect2.tl().y);
	int maxx = min(rect1.br().x, rect2.br().x);
	int maxy = min(rect1.br().y, rect2.br().y);

	return Rect(Point(minx,miny), Point(maxx, maxy));
}

void nms_pickDmtxRects(vector<vector<rectBRConfience>> &dmtxRects)
{
	// the svm predict fun ret val(maybe negative), closer to 0 means better confidence?
	//ＰＡＳＣＡＬtermcriteria
	/*	 Ａｒｅａ（ＢＢｇｔ∩ＢＢｄｔ）／Ａｒｅａ（ＢＢｇｔ∪ＢＢｄｔ）> 0.5?*/
	for (int j = 0;j< static_cast<int>(dmtxRects.size());j++)
	{

		if (dmtxRects[j].size() == 1)
		{
			dmtxRects[j][0].ifMax = true;
		}

		else
		{
			rectBRConfience mostConfiRect;
			int mostK = 0;
			mostConfiRect.svmPreConfidence = 0;

			//get the most confident rect
			for (int k = 0 ; k < static_cast<int>(dmtxRects[j].size());k++)
			{
				if (dmtxRects[j][k].svmPreConfidence < mostConfiRect.svmPreConfidence)
				{
					mostConfiRect = dmtxRects[j][k];
					mostK = k;
				}
			}//for (int k = 0 ; k < dmtxRects[j].size();k++)

			//mark this most confidence one as true for suppress
			dmtxRects[j][mostK].ifMax = true;

			//do suppress
			for (int k = 0 ; k < static_cast<int>(dmtxRects[j].size());k++)
			{
				if (k == mostK)
				{
					continue;
				}

				double overlapArea = 
					(double)Rect1OverlapsRect2(dmtxRects[j][mostK].dmtx, dmtxRects[j][k].dmtx);
				if (overlapArea / 
					(dmtxRects[j][mostK].dmtx.area() + dmtxRects[j][k].dmtx.area() - overlapArea) > 0.5)
				{//this says, this k is a good predict. merge this to mostK
					dmtxRects[j][mostK].dmtx = 
						Rect1PlusRect2(dmtxRects[j][mostK].dmtx, dmtxRects[j][k].dmtx);
				}
			}//for (int k = 0 ; k < dmtxRects[j].size();k++)

		}//else..if (dmtxRects[j].size() == 1)

	}//for (int j = 0;j<dmtxRects.size();j++)

}

/**************************************************************************************************************************/
/*  获取文件夹下所有文件名
输入：		
path	:	文件夹路径
exd		:   所要获取的文件名后缀，如jpg、png等；如果希望获取所有
文件名, exd = ""
输出：
files	:	获取的文件名列表
*/
/**************************************************************************************************************************/
void getFiles( string path, string exd, vector<string>& files )
{
	//文件句柄
	long   hFile   =   0;
	//文件信息
	struct _finddata_t fileinfo;
	string pathName, exdName;

	if (0 != strcmp(exd.c_str(), ""))
	{
		exdName = "\\*." + exd;
	}
	else
	{
		exdName = "\\*";
	}

	if((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(),&fileinfo)) !=  -1)
	{
		do
		{
			//如果是文件夹中仍有文件夹,迭代之
			//如果不是,加入列表
			if((fileinfo.attrib &  _A_SUBDIR))
			{
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
					getFiles( pathName.assign(path).append("\\").append(fileinfo.name), exd, files );
			}
			else
			{
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
					files.push_back(pathName.assign(path).append("\\").append(fileinfo.name));
			}
		}while(_findnext(hFile, &fileinfo)  == 0);
		_findclose(hFile);
	}
}
