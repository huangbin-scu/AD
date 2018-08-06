#ifndef PIXEL_DENSITY_H
#define PIXEL_DENSITY_H

#include <highgui.h> 
#include <cv.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>
using namespace std;

#define POINT_NUM_PER_PIXEL 0.05 //���ڱ�Ե������ֵ����ȡ
#define JUDGE_FRAME_PIXEL 20 //�ܶȵȼ����ж�֡��

class CPixelDensity
{
public:
	CPixelDensity();

	void InitialPixelDensity(IplImage* pFrm);
	char* DensityEstimate(IplImage* pFrm);

	~CPixelDensity();

private:
	IplImage* m_pPreEdge; //ǰһ֡ͼ���Ե
	char* m_text; //��¼�ܶȵȼ�
	int m_pedestrian; //���������˸���
	ofstream m_outfile; //�������������Ϣ���ļ�
	vector<bool> m_preFramePixel; //���ڴ洢��ǰ֡���ܶȵȼ�
};

#endif