#include "ViBeDetect.h"

static int c_xoff[9] = {-1,  0,  1, -1, 1, -1, 0, 1, 0};//x的邻居点
static int c_yoff[9] = {-1,  0,  1, -1, 1, -1, 0, 1, 0};//y的邻居点

CViBeDetect::CViBeDetect()
{

}

/* 初始化
 * pGrayMat[in]：待处理的灰度图像
 */
void CViBeDetect::InitialDetect(CvMat* pGrayMat)
{
	//创建一个随机数生成器
	cv::RNG rng(0xFFFFFFFF);

	//记录随机生成的 行(r) 和 列(c)
	int rand, r, c;

	//对每个像素样本进行初始化
	for (int i = 0; i < pGrayMat->rows; i++)
	{
		for (int j = 0; j < pGrayMat->cols; j++)
		{
			for (int k = 0; k < defaultNbSamples; k++)
			{
				//随机获取像素样本值
				rand = rng.uniform(0, 9);

				r = i + c_yoff[rand]; 
				if (r < 0) 
					r = 0; 
				if (r >= pGrayMat->rows) 
					r = pGrayMat->rows - 1;	//行

				c = j + c_xoff[rand]; 
				if (c < 0) 
					c = 0; 
				if (c >= pGrayMat->cols) 
					c = pGrayMat->cols - 1;	//列

				//存储像素样本值
				m_samples[i][j][k] = CV_MAT_ELEM(*pGrayMat, float, r, c);
			}
			m_samples[i][j][defaultNbSamples] = 0;
		}
	}
}


/* 更新
 * pImage[out]：经过目标检测后的二值图像
 * nFrmNum[in]：当前帧的帧数
 */
int CViBeDetect::update(CvMat* pGrayMat, CvMat* pSegMat, int nFrmNum)
{
	//创建一个随机数生成器
	cv::RNG rng(0xFFFFFFFF);

	int foregroundPixels = 0;
	for (int i = 0; i < pGrayMat->rows; i++)
	{	
		for (int j = 0; j < pGrayMat->cols; j++)
		{

			//用于判断一个点是否是背景点,index记录已比较的样本个数，count表示匹配的样本个数
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
				//判断为背景像素,只有背景点才能被用来传播和更新存储样本值
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
				//判断为前景像素
				CV_MAT_ELEM(*pSegMat, float, i, j) = foreground;
				foregroundPixels ++;
			}
		}
	}
	return foregroundPixels;
}