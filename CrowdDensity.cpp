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

/* ����ViBe�㷨����Ŀ����
 * ����Ƶ��ǰJUDGE_FRAME_DETECT֡����Ŀ���⣬ȡǰ������ռ����ͼ��ı����ľ�ֵ��Ϊ����Ƶ��ǰ������ռ����ͼ��ı���
 * pImage[in]��������ĵ�ǰ֡��ͼ��
 * nFrmNum[in]����ǰ֡��֡��
 */
void CCrowdDensity::VedioDetect(IplImage* pImage, int nFrmNum)
{
	//��Ϊ��0֡ʱ��ʼ��������
	if (nFrmNum == 0)
	{
		m_pGrayMat=cvCreateMat(pImage->height, pImage->width, CV_32FC1);
		m_pGrayImage = cvCreateImage(cvSize(pImage->width, pImage->height), IPL_DEPTH_8U, 1);
		m_pSegMat = cvCreateMat(pImage->height, pImage->width, CV_32FC1);
		m_pSegImage = cvCreateImage(cvSize(pImage->width, pImage->height), IPL_DEPTH_8U, 1);

		//ת���ɵ�ͨ��ͼ���ٴ���
		cvCvtColor(pImage, m_pGrayImage, CV_RGB2GRAY);
		cvConvert(m_pGrayImage, m_pGrayMat);
		m_pViBe->InitialDetect(m_pGrayMat);
	}
	//ͨ����1֡����JUDGE_FRAME_DETECT֡����ǰ������ռ����ͼ��ľ�ֵ
	else 
	{
		cvCvtColor(pImage, m_pGrayImage, CV_RGB2GRAY);
		cvConvert(m_pGrayImage, m_pGrayMat);
		int forePixels = m_pViBe->update(m_pGrayMat, m_pSegMat, nFrmNum); //���±���
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

/* ��ʼ���������ص��ܶȹ��ƺͻ���������ܶȹ��ƺ�ѵ��
 * pImage[in]��������ĵ�ǰ֡��ͼ��
 */
void CCrowdDensity::InitialCrowdDensity(IplImage* pImage)
{
	m_pPixelDensity->InitialPixelDensity(pImage);
	m_pTextureDensity->InitialTextureDensity(TEST_PROCESS);
}

/* �ж��ܶȵȼ�
 * ��ǰ������ռ����ͼ��ı���С��DETECT_THRESHOLD�����û������صķ�����������û�������ķ���
 * pImage[in]��������ĵ�ǰ֡��ͼ��
 * nFrmNum[in]����ǰ֡��֡��
 */
char* CCrowdDensity::DesityRank(IplImage* pImage, int nFrmNum)
{
	//��֡��С�ڵ���JUDGE_FRAME_DETECT��Ŀ����
	if (nFrmNum < JUDGE_FRAME_DETECT && detectFlag)
		VedioDetect(pImage, nFrmNum);

	//��֡������JUDGE_FRAME_DETECT������DETECT_THRESHOLDѡ���ܶȹ��Ʒ������ж��ܶȵȼ�
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