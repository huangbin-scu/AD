#ifndef VIBE_DETECT_H
#define VIBE_DETECT_H

#include "highgui.h"
#include "cv.h"
#include <iostream>
#include <math.h>

using namespace std;

#define defaultNbSamples 20 //ÿ�����ص����������
#define defaultReqMatches 2 //#minָ��
#define defaultRadius 20 //Sqthere�뾶
#define defaultSubsamplingFactor 16 //�Ӳ�������
#define background 0 //��������
#define foreground 255 //ǰ������

class CViBeDetect
{
public:
	CViBeDetect();
	void InitialDetect(CvMat* pGrayMat);
	int update(CvMat* pGrayMat, CvMat* pSegMat, int nFrmNum);

private:
	float m_samples[1024][1024][defaultNbSamples + 1]; //����ÿ�����ص������ֵ
};

#endif