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

#define CHARACTER_NUM 20 //������Ŀ
#define HIGH_IMAGE_NUM 713 //���ܶȵ���������
#define VERY_HIGH_IMAGE_NUM 714 //�����ܶȵ���������

#define JUDGE_FRAME_TEXTURE 15 //�ܶȵȼ����ж�֡��

#define TRAIN_PROCESS 0//ѵ����־
#define TEST_PROCESS 1//���Ա�־

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
	int m_FilterWindowWidth; //���������Ĵ�С��ͨ����ͼ�񻮷ֳ����ɸ���������
	int m_distance;	//���룬���ݲ�ͬ��Ӧ��ѡȡ	
	int m_GrayLayerNum;	//�Ҷȷּ�

	int** m_PMatrixH; //0�ȷ����ϵĻҶȹ��־���
	int** m_PMatrixRD; //45�ȷ����ϵĻҶȹ��־���
	int** m_PMatrixV; //90�ȷ����ϵĻҶȹ��־���
	int** m_PMatrixLD; //135�ȷ����ϵĻҶȹ��־���

	int m_svmResponse[HIGH_IMAGE_NUM + VERY_HIGH_IMAGE_NUM]; //svm�������Ķ�Ӧ��������
	double m_svmData[(HIGH_IMAGE_NUM + VERY_HIGH_IMAGE_NUM) * CHARACTER_NUM]; //svm����������������
	double m_testData[CHARACTER_NUM]; //����ͼƬ��������

	CvMat* m_svmDataMat; //��������������
	CvMat* m_svmResponseMat; //�����������������
	CvMat* m_testMat; //����ͼƬ��������

	CvTermCriteria m_criteria;
	CvSVMParams m_param;
	CvSVM m_svm; //������

	vector<bool> m_preFrameTexture; //��ǰ֡��ͼ����ܶȵȼ�
};

#endif