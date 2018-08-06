#ifndef VIBE_DETECT_H
#define VIBE_DETECT_H

#include "highgui.h"
#include "cv.h"
#include <iostream>
#include <math.h>

using namespace std;

#define defaultNbSamples 20 //每个像素点的样本个数
#define defaultReqMatches 2 //#min指数
#define defaultRadius 20 //Sqthere半径
#define defaultSubsamplingFactor 16 //子采样概率
#define background 0 //背景像素
#define foreground 255 //前景像素

class CViBeDetect
{
public:
	CViBeDetect();
	void InitialDetect(CvMat* pGrayMat);
	int update(CvMat* pGrayMat, CvMat* pSegMat, int nFrmNum);

private:
	float m_samples[1024][1024][defaultNbSamples + 1]; //保存每个像素点的样本值
};

#endif