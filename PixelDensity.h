#ifndef PIXEL_DENSITY_H
#define PIXEL_DENSITY_H

#include <highgui.h> 
#include <cv.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>
using namespace std;

#define POINT_NUM_PER_PIXEL 0.05 //用于边缘点数阈值的求取
#define JUDGE_FRAME_PIXEL 20 //密度等级的判断帧数

class CPixelDensity
{
public:
	CPixelDensity();

	void InitialPixelDensity(IplImage* pFrm);
	char* DensityEstimate(IplImage* pFrm);

	~CPixelDensity();

private:
	IplImage* m_pPreEdge; //前一帧图像边缘
	char* m_text; //记录密度等级
	int m_pedestrian; //检测出的行人个数
	ofstream m_outfile; //用于输出特征信息到文件
	vector<bool> m_preFramePixel; //用于存储以前帧的密度等级
};

#endif