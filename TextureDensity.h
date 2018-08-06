#ifndef TEXTURE_DENSITY_H
#define TEXTURE_DENSITY_H

#include <iostream>
#include <highgui.h>
#include <cv.h>
#include <math.h>
#include <ml.h>
#include <cvaux.h>
#include <fstream>
#include <vector>
using namespace std;

#define CHARACTER_NUM 20 //特征数目
#define HIGH_IMAGE_NUM 713 //高密度的样本个数
#define VERY_HIGH_IMAGE_NUM 714 //超高密度的样本个数

#define JUDGE_FRAME_TEXTURE 15 //密度等级的判断帧数

#define TRAIN_PROCESS 0//训练标志
#define TEST_PROCESS 1//测试标志

class CTextureDensity
{
public:
	CTextureDensity();

	void InitialTextureDensity(int flag);
	void TrainSample();
	void CollectCharacter(IplImage* pImage, double Feature[]);
	void StoreCharacter(IplImage* pImage, double* pData);

	void ComputeMatrix(IplImage* LocalImage);
	void ComputeFeature(double& FeatureEnergy, double& FeatureEntropy, double& FeatureInertiaQuadrature, 
		double& FeatureCorrelation, double& FeatureLocalCalm, int MatrixDirection);

	char* TestVedio(IplImage* pImage);

	~CTextureDensity();

private:
	int m_FilterWindowWidth; //纹理区域块的大小，通常将图像划分成若干个纹理块计算
	int m_distance;	//距离，依据不同的应用选取	
	int m_GrayLayerNum;	//灰度分级

	int** m_PMatrixH; //0度方向上的灰度共现矩阵
	int** m_PMatrixRD; //45度方向上的灰度共现矩阵
	int** m_PMatrixV; //90度方向上的灰度共现矩阵
	int** m_PMatrixLD; //135度方向上的灰度共现矩阵

	int m_svmResponse[HIGH_IMAGE_NUM + VERY_HIGH_IMAGE_NUM]; //svm中样本的对应类别的数组
	double m_svmData[(HIGH_IMAGE_NUM + VERY_HIGH_IMAGE_NUM) * CHARACTER_NUM]; //svm中样本特征的数组
	double m_testData[CHARACTER_NUM]; //测试图片特征数组

	CvMat* m_svmDataMat; //分类器特征矩阵
	CvMat* m_svmResponseMat; //分类器样本分类矩阵
	CvMat* m_testMat; //测试图片特征矩阵

	CvTermCriteria m_criteria;
	CvSVMParams m_param;
	CvSVM m_svm; //分类器

	vector<bool> m_preFrameTexture; //以前帧的图像的密度等级
};

#endif