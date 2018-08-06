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

#define JUDGE_FRAME_DETECT 50 //根据视频的前JUDGE_FRAME_DETECT帧进行目标检测，确定密度统计采用的方法
#define DETECT_THRESHOLD 0.34 //目标检测的前景比例判断阈值，当大于该值时采用基于纹理的方法，当小于该值时采用基于像素的方法

class CCrowdDensity
{
public:
	CCrowdDensity();

	void VedioDetect(IplImage* pImage, int nfrmNum);
	void InitialCrowdDensity(IplImage* pImage);
	char* DesityRank(IplImage* pImage, int nFrmNum);

	~CCrowdDensity();

private:
	CViBeDetect* m_pViBe; //目标检测的对象
	IplImage* m_pGrayImage; //保存pFrame对应的灰度图像
	IplImage* m_pSegImage; //保存目标检测后的二值图像
	CvMat* m_pSegMat; //二值图像的矩阵
	CvMat* m_pGrayMat; //灰度图像的矩阵
    double m_preForePixelsPro[JUDGE_FRAME_DETECT - 1]; //前JUDGE_FRAME_DETECT-1帧的每帧前景像素占整幅图像的比例
	double m_meanForePixelPro; //JUDGE_FRAME_DETECT帧的前景像素占整幅图像的比例的均值
	char* m_text; //等级标记

	CTextureDensity* m_pTextureDensity; //基于纹理的密度估计的对象
	CPixelDensity* m_pPixelDensity; //基于像素的密度估计的对象
};

#endif