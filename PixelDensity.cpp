#include "PixelDensity.h"

CPixelDensity::CPixelDensity()
{
	m_preFramePixel.clear();
}

/* ��ʼ���������ص��ܶȹ��ƣ���ȡ��һ֡�ı�Ե
 * pFrm[in]��������ĵ�ǰ֡��ͼ��
 */
void CPixelDensity::InitialPixelDensity(IplImage* pFrm)
{
	m_pPreEdge = cvCreateImage(cvSize(pFrm->width, pFrm->height), IPL_DEPTH_8U, 1); 

	IplImage* pCurrentEdge = cvCreateImage(cvSize(pFrm->width, pFrm->height), IPL_DEPTH_8U, 1); 	
	IplImage* pGrayImg = cvCreateImage(cvSize(pFrm->width, pFrm->height), IPL_DEPTH_8U, 1);

	cvCvtColor(pFrm, pGrayImg, CV_BGR2GRAY);
	cvSmooth(pGrayImg, pGrayImg, CV_GAUSSIAN, 3, 3, 0);
	cvCanny(pGrayImg, m_pPreEdge, 80.0, 80.0 * 3, 3);
}

/* ����ͼ��ı�Ե��Ϣ���������������ܶ�
 * pFrm[in]��������ĵ�ǰ֡��ͼ��
 */
char* CPixelDensity::DensityEstimate(IplImage* pFrm)
{
	m_pedestrian = 0;
	int rect_w, rect_h; //�������ڴ�С
	int PN; //���ص�����ֵ

	IplImage* pCurrentEdge = cvCreateImage(cvSize(pFrm->width, pFrm->height), IPL_DEPTH_8U, 1); //��ǰ֡ͼ��ı�Ե
	IplImage* pDiffEdge = cvCreateImage(cvSize(pFrm->width, pFrm->height), IPL_DEPTH_8U, 1); //��ǰ֡ͼ���Ե��ǰһ֡ͼ���Ե�Ĳ��
	IplImage* pTrueEdge = pTrueEdge = cvCreateImage(cvSize(pFrm->width, pFrm->height), IPL_DEPTH_8U, 1); //��ǰ֡����ʵͼ���Ե
	IplImage* pGrayImg = cvCreateImage(cvSize(pFrm->width, pFrm->height), IPL_DEPTH_8U, 1); //�Ҷ�ͼ��

	//clock_t TimeStart = clock();					//��ʼ��ʱ
			
	cvCvtColor(pFrm, pGrayImg, CV_BGR2GRAY); //ת���ɵ�ͨ��ͼ���ٴ���
	cvSmooth(pGrayImg, pGrayImg, CV_GAUSSIAN, 3, 3, 0);
	cvCanny(pGrayImg, pCurrentEdge, 80.0, 80.0 * 3, 3);

	cvAbsDiff(pCurrentEdge, m_pPreEdge, pDiffEdge); //��ǰ֡������ͼ���(�󱳾��ȡ����ֵ)
	cvSmooth(pDiffEdge, pDiffEdge, CV_GAUSSIAN, 3, 3, 0);
	cvDilate(pDiffEdge, pDiffEdge, 0, 1);
	cvErode(pDiffEdge, pDiffEdge, 0, 1); //������̬ѧ�˲���ȥ������

	for (int i = 0; i < pTrueEdge->height; i++)
	{
		for (int j = 0; j < pTrueEdge->width; j++)
			CV_IMAGE_ELEM(pTrueEdge, unsigned char, i, j) = 0;
	}
	cvCopy(pCurrentEdge, pTrueEdge, pDiffEdge);

	CvMemStorage* storage1 = cvCreateMemStorage();
	CvSeq* HypoRect = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvRect), storage1);
	CvSeqWriter seqW;
	CvSeqReader seqR;
	
	cvStartAppendToSeq(HypoRect, &seqW);
		
	//��һ��ͼ�񻮷ֳɺܶ�����ο򣬼���ÿ�����ο��еı�Ե���ص���
	for (int x = 0;x < (pTrueEdge->width - 39); x += (0.0215 * x + 24.3369) * 0.13333 + 4.8)
	{				
		rect_w = 0.0215 * x + 24.3369;
		rect_h = rect_w * 2.7;
		IplImage * img = cvCreateImage(cvSize(rect_w, rect_h), IPL_DEPTH_8U, 1);
		PN = POINT_NUM_PER_PIXEL * rect_w * rect_h;

		for (int y = 0; y < (pTrueEdge->height - 105); y += rect_w * 0.13333 + 4.8)
		{
			CvRect nRect(cvRect(x, y, rect_w, rect_h));
			cvSetImageROI(pTrueEdge, nRect);
			cvCopy(pTrueEdge, img);
			cvResetImageROI(pTrueEdge);
					
			CvMemStorage* storage = cvCreateMemStorage();
			CvSeq* contour = 0;
			cvFindContours(img, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
			CvSeq *pCurSeq = contour;
			int ntotal = 0;
			while (pCurSeq)  
			{  
				ntotal += pCurSeq->total;
				pCurSeq = pCurSeq->h_next;
			}
			if(ntotal > PN) //���ݵ����Ŀ��ɸѡ
			{
				CV_WRITE_SEQ_ELEM(nRect, seqW);
			}
			cvReleaseMemStorage(&storage);
		}
		cvReleaseImage(&img);
	}
	cvEndWriteSeq(&seqW);

	float PeoN = 0; //����������ֵ		
	PeoN = 0.0385 * HypoRect->total + 1.7765; //ֱ����Ϸ��̣���Ե���ص����������Ĺ�ϵ
	m_pedestrian = static_cast<int>(PeoN + 0.5);

	///////////////////////////////////////////////�����ο�

	//cvStartReadSeq(HypoRect,&seqR,0);
	//for (int i = 0; i < HypoRect->total; i++)
	//{
	//	CvRect nRect;
	//	CV_READ_SEQ_ELEM(nRect, seqR);
	//	cvRectangle(pFrm, cvPoint(nRect.x, nRect.y), 
	//              cvPoint(nRect.x + nRect.width, nRect.y + nRect.height), cvScalar(255), 1, 8, 0);
	//}
	//std::cout << nFrmNum << "\t" << PN << "\t" << PeoN << std::endl;

///////////////////////////////////////////////����ܶȵȼ�

	//����ͳ�Ƶ���ǰ֡���ܶȵȼ�������ǰ֡���ܶȵȼ�
	bool currentRank;
	if (m_pedestrian < 10)
		currentRank = false;
	else 
		currentRank = true;
	m_preFramePixel.push_back(currentRank);

	int pixelLow = 0, pixelMedium = 0;
	if (m_preFramePixel.size() < JUDGE_FRAME_PIXEL)
	{
		for (vector<bool>::iterator iter = m_preFramePixel.begin(); iter != m_preFramePixel.end(); ++iter)
		{
			if (*iter == true)
				pixelMedium++;
			else
				pixelLow++;
		}
	}
	else
	{
		for (vector<bool>::iterator iter = m_preFramePixel.end() - JUDGE_FRAME_PIXEL; iter != m_preFramePixel.end(); ++iter)
		{
			if (*iter == true)
				pixelMedium++;
			else
				pixelLow++;
		}
	}
	if (pixelLow > pixelMedium)
		m_text = "Level: Low";
	else
		m_text = "Level: Medium";

	//clock_t TimeEnd = clock();
	//clock_t CostTime = TimeEnd - TimeStart;
	//cout << CostTime << endl;

	cvCopy(pCurrentEdge, m_pPreEdge,0);

	cvReleaseImage(&pCurrentEdge);
	cvReleaseImage(&pDiffEdge);
	cvReleaseImage(&pTrueEdge);
	cvReleaseImage(&pGrayImg);

	return m_text;
} 

CPixelDensity::~CPixelDensity()
{
	cvDestroyAllWindows();
	cvReleaseImage(&m_pPreEdge);
}