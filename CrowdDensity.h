#ifndef CROWD_DENSITY_H
#define CROWD_DENSITY_H

#include <iostream>
#include <highgui.h>
#include <cv.h>
#include <math.h>
#include <ml.h>
#include <cvaux.h>
#include <fstream>
#include "ViBeDetect.h"
#include "PixelDensity.h" 
#include "TextureDensity.h"
using namespace std;

#define JUDGE_FRAME_DETECT 50 //������Ƶ��ǰJUDGE_FRAME_DETECT֡����Ŀ���⣬ȷ���ܶ�ͳ�Ʋ��õķ���
#define DETECT_THRESHOLD 0.34 //Ŀ�����ǰ�������ж���ֵ�������ڸ�ֵʱ���û�������ķ�������С�ڸ�ֵʱ���û������صķ���

class CCrowdDensity
{
public:
	CCrowdDensity();

	void VedioDetect(IplImage* pImage, int nfrmNum);
	void InitialCrowdDensity(IplImage* pImage);
	char* DesityRank(IplImage* pImage, int nFrmNum);

	~CCrowdDensity();

private:
	CViBeDetect* m_pViBe; //Ŀ����Ķ���
	IplImage* m_pGrayImage; //����pFrame��Ӧ�ĻҶ�ͼ��
	IplImage* m_pSegImage; //����Ŀ�����Ķ�ֵͼ��
	CvMat* m_pSegMat; //��ֵͼ��ľ���
	CvMat* m_pGrayMat; //�Ҷ�ͼ��ľ���
    double m_preForePixelsPro[JUDGE_FRAME_DETECT - 1]; //ǰJUDGE_FRAME_DETECT-1֡��ÿ֡ǰ������ռ����ͼ��ı���
	double m_meanForePixelPro; //JUDGE_FRAME_DETECT֡��ǰ������ռ����ͼ��ı����ľ�ֵ
	char* m_text; //�ȼ����

	CTextureDensity* m_pTextureDensity; //����������ܶȹ��ƵĶ���
	CPixelDensity* m_pPixelDensity; //�������ص��ܶȹ��ƵĶ���
};

#endif