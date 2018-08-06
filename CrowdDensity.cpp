#include "CrowdDensity.h"
extern bool detectFlag;

CCrowdDensity::CCrowdDensity()
{
	m_pViBe = new CViBeDetect();
	m_pPixelDensity = new CPixelDensity();
	m_pTextureDensity = new CTextureDensity();
	m_meanForePixelPro = 0.0;
	m_text = NULL;
}

/* 利用ViBe算法进行目标检测
 * 对视频的前JUDGE_FRAME_DETECT帧进行目标检测，取前景像素占整幅图像的比例的均值作为该视频的前景像素占整幅图像的比例
 * pImage[in]：待处理的当前帧的图像
 * nFrmNum[in]：当前帧的帧数
 */
void CCrowdDensity::VedioDetect(IplImage* pImage, int nFrmNum)
{
	//当为第0帧时初始化检测参数
	if (nFrmNum == 0)
	{
		m_pGrayMat=cvCreateMat(pImage->height, pImage->width, CV_32FC1);
		m_pGrayImage = cvCreateImage(cvSize(pImage->width, pImage->height), IPL_DEPTH_8U, 1);
		m_pSegMat = cvCreateMat(pImage->height, pImage->width, CV_32FC1);
		m_pSegImage = cvCreateImage(cvSize(pImage->width, pImage->height), IPL_DEPTH_8U, 1);

		//转化成单通道图像再处理
		cvCvtColor(pImage, m_pGrayImage, CV_RGB2GRAY);
		cvConvert(m_pGrayImage, m_pGrayMat);
		m_pViBe->InitialDetect(m_pGrayMat);
	}
	//通过第1帧到第JUDGE_FRAME_DETECT帧计算前景像素占整幅图像的均值
	else 
	{
		cvCvtColor(pImage, m_pGrayImage, CV_RGB2GRAY);
		cvConvert(m_pGrayImage, m_pGrayMat);
		int forePixels = m_pViBe->update(m_pGrayMat, m_pSegMat, nFrmNum); //更新背景
		cvConvert(m_pSegMat, m_pSegImage);
		double currentForePixelsPro = (double)forePixels / (1.0 * m_pSegImage->height * m_pSegImage->width);

		m_preForePixelsPro[nFrmNum - 1] = currentForePixelsPro;
		if (nFrmNum == JUDGE_FRAME_DETECT - 1)
		{
			for (int i = 0; i < JUDGE_FRAME_DETECT - 1; i++)
			{
				m_meanForePixelPro += m_preForePixelsPro[i];
			}
			m_meanForePixelPro /= (JUDGE_FRAME_DETECT - 1);
		}
	}
}

/* 初始化基于像素的密度估计和基于纹理的密度估计和训练
 * pImage[in]：待处理的当前帧的图像
 */
void CCrowdDensity::InitialCrowdDensity(IplImage* pImage)
{
	m_pPixelDensity->InitialPixelDensity(pImage);
	m_pTextureDensity->InitialTextureDensity(TEST_PROCESS);
}

/* 判断密度等级
 * 当前景像素占整幅图像的比例小于DETECT_THRESHOLD，采用基于像素的方法，否则采用基于纹理的方法
 * pImage[in]：待处理的当前帧的图像
 * nFrmNum[in]：当前帧的帧数
 */
char* CCrowdDensity::DesityRank(IplImage* pImage, int nFrmNum)
{
	//当帧数小于等于JUDGE_FRAME_DETECT，目标检测
	if (nFrmNum < JUDGE_FRAME_DETECT && detectFlag)
		VedioDetect(pImage, nFrmNum);

	//当帧数大于JUDGE_FRAME_DETECT，根据DETECT_THRESHOLD选择密度估计方法，判断密度等级
	else
	{
		if (m_meanForePixelPro < DETECT_THRESHOLD)
			m_text = m_pPixelDensity->DensityEstimate(pImage);
		else
			m_text = m_pTextureDensity->TestVedio(pImage);
	}
	return m_text;
}

CCrowdDensity::~CCrowdDensity()
{
	delete m_pViBe;
	delete m_pPixelDensity;
	delete m_pTextureDensity;
}