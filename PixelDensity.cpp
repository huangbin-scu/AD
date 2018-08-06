#include "PixelDensity.h"

CPixelDensity::CPixelDensity()
{
	m_preFramePixel.clear();
}

/* 初始化基于像素的密度估计，求取第一帧的边缘
 * pFrm[in]：待处理的当前帧的图像
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

/* 利用图像的边缘信息计算人数，估计密度
 * pFrm[in]：待处理的当前帧的图像
 */
char* CPixelDensity::DensityEstimate(IplImage* pFrm)
{
	m_pedestrian = 0;
	int rect_w, rect_h; //滑动窗口大小
	int PN; //像素点数阈值

	IplImage* pCurrentEdge = cvCreateImage(cvSize(pFrm->width, pFrm->height), IPL_DEPTH_8U, 1); //当前帧图像的边缘
	IplImage* pDiffEdge = cvCreateImage(cvSize(pFrm->width, pFrm->height), IPL_DEPTH_8U, 1); //当前帧图像边缘与前一帧图像边缘的差分
	IplImage* pTrueEdge = pTrueEdge = cvCreateImage(cvSize(pFrm->width, pFrm->height), IPL_DEPTH_8U, 1); //当前帧的真实图像边缘
	IplImage* pGrayImg = cvCreateImage(cvSize(pFrm->width, pFrm->height), IPL_DEPTH_8U, 1); //灰度图像

	//clock_t TimeStart = clock();					//开始计时
			
	cvCvtColor(pFrm, pGrayImg, CV_BGR2GRAY); //转化成单通道图像再处理
	cvSmooth(pGrayImg, pGrayImg, CV_GAUSSIAN, 3, 3, 0);
	cvCanny(pGrayImg, pCurrentEdge, 80.0, 80.0 * 3, 3);

	cvAbsDiff(pCurrentEdge, m_pPreEdge, pDiffEdge); //当前帧跟背景图相减(求背景差并取绝对值)
	cvSmooth(pDiffEdge, pDiffEdge, CV_GAUSSIAN, 3, 3, 0);
	cvDilate(pDiffEdge, pDiffEdge, 0, 1);
	cvErode(pDiffEdge, pDiffEdge, 0, 1); //进行形态学滤波，去掉噪音

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
		
	//将一幅图像划分成很多个矩形框，计算每个矩形框中的边缘像素点数
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
			if(ntotal > PN) //根据点的数目做筛选
			{
				CV_WRITE_SEQ_ELEM(nRect, seqW);
			}
			cvReleaseMemStorage(&storage);
		}
		cvReleaseImage(&img);
	}
	cvEndWriteSeq(&seqW);

	float PeoN = 0; //行人数估计值		
	PeoN = 0.0385 * HypoRect->total + 1.7765; //直线拟合方程，边缘像素点数与人数的关系
	m_pedestrian = static_cast<int>(PeoN + 0.5);

	///////////////////////////////////////////////画矩形框

	//cvStartReadSeq(HypoRect,&seqR,0);
	//for (int i = 0; i < HypoRect->total; i++)
	//{
	//	CvRect nRect;
	//	CV_READ_SEQ_ELEM(nRect, seqR);
	//	cvRectangle(pFrm, cvPoint(nRect.x, nRect.y), 
	//              cvPoint(nRect.x + nRect.width, nRect.y + nRect.height), cvScalar(255), 1, 8, 0);
	//}
	//std::cout << nFrmNum << "\t" << PN << "\t" << PeoN << std::endl;

///////////////////////////////////////////////标记密度等级

	//根据统计的以前帧的密度等级给出当前帧的密度等级
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