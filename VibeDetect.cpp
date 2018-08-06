#include "ViBeDetect.h"

static int c_xoff[9] = {-1,  0,  1, -1, 1, -1, 0, 1, 0};//x���ھӵ�
static int c_yoff[9] = {-1,  0,  1, -1, 1, -1, 0, 1, 0};//y���ھӵ�

CViBeDetect::CViBeDetect()
{

}

/* ��ʼ��
 * pGrayMat[in]��������ĻҶ�ͼ��
 */
void CViBeDetect::InitialDetect(CvMat* pGrayMat)
{
	//����һ�������������
	cv::RNG rng(0xFFFFFFFF);

	//��¼������ɵ� ��(r) �� ��(c)
	int rand, r, c;

	//��ÿ�������������г�ʼ��
	for (int i = 0; i < pGrayMat->rows; i++)
	{
		for (int j = 0; j < pGrayMat->cols; j++)
		{
			for (int k = 0; k < defaultNbSamples; k++)
			{
				//�����ȡ��������ֵ
				rand = rng.uniform(0, 9);

				r = i + c_yoff[rand]; 
				if (r < 0) 
					r = 0; 
				if (r >= pGrayMat->rows) 
					r = pGrayMat->rows - 1;	//��

				c = j + c_xoff[rand]; 
				if (c < 0) 
					c = 0; 
				if (c >= pGrayMat->cols) 
					c = pGrayMat->cols - 1;	//��

				//�洢��������ֵ
				m_samples[i][j][k] = CV_MAT_ELEM(*pGrayMat, float, r, c);
			}
			m_samples[i][j][defaultNbSamples] = 0;
		}
	}
}


/* ����
 * pImage[out]������Ŀ�����Ķ�ֵͼ��
 * nFrmNum[in]����ǰ֡��֡��
 */
int CViBeDetect::update(CvMat* pGrayMat, CvMat* pSegMat, int nFrmNum)
{
	//����һ�������������
	cv::RNG rng(0xFFFFFFFF);

	int foregroundPixels = 0;
	for (int i = 0; i < pGrayMat->rows; i++)
	{	
		for (int j = 0; j < pGrayMat->cols; j++)
		{

			//�����ж�һ�����Ƿ��Ǳ�����,index��¼�ѱȽϵ�����������count��ʾƥ�����������
			int count = 0,index = 0;
			float dist = 0;
			//
			while((count < defaultReqMatches) && (index < defaultNbSamples))
			{
				dist = CV_MAT_ELEM(*pGrayMat, float, i, j) - m_samples[i][j][index];
				if (dist < 0)
					dist = -dist;

				if (dist < defaultRadius) 
					count++;
				index++;
			}
			if (count >= defaultReqMatches)
			{
				//�ж�Ϊ��������,ֻ�б�������ܱ����������͸��´洢����ֵ
				m_samples[i][j][defaultNbSamples] = 0;

				CV_MAT_ELEM(*pSegMat, float, i, j) = background;

				int rand = rng.uniform(0, defaultSubsamplingFactor);
				if (rand == 0)
				{
					rand = rng.uniform(0, defaultNbSamples);
					m_samples[i][j][rand] = CV_MAT_ELEM(*pGrayMat, float, i, j);
				}
				rand = rng.uniform(0, defaultSubsamplingFactor);
				if (rand == 0)
				{
					int xN,yN;

					rand = rng.uniform(0, 9);
					yN = i + c_yoff[rand];
					if (yN < 0) 
						yN = 0; 
					if (yN >= pGrayMat->height) 
						yN = pGrayMat->height - 1;

					rand = rng.uniform(0,9);
					xN = j + c_xoff[rand];
					if (xN < 0) 
						xN = 0; 
					if (xN >= pGrayMat->width)
						xN = pGrayMat->width - 1;

					rand = rng.uniform(0, defaultNbSamples);
					m_samples[yN][xN][rand] = CV_MAT_ELEM(*pGrayMat, float, i, j);
				} 
			}
			else 
			{
				//�ж�Ϊǰ������
				CV_MAT_ELEM(*pSegMat, float, i, j) = foreground;
				foregroundPixels ++;
			}
		}
	}
	return foregroundPixels;
}