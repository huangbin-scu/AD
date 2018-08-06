/*
 *�ò��ָ������Ƶ���ù�����ֵ�仯����������
*/

#include "vif.h"
#include <numeric>
#include <ctime>

/*
 *@brief ������Ƶ  globle descriptorȫ��������ʹ��vif������Ϊ��Ƶ��ȫ������
 *@param[in]video ��Ƶ����
 *@param[out]desc �õ�����������
*/
void VIF::descripVideoG(const std::vector<cv::Mat>& video, cv::Mat& desc)
{
	//��ȡ����ģֵ
	std::vector<cv::Mat> optFlowMagnitude;

	IplImage* velx, *vely, *pCurrGray, *pPreGray;
	
	for (int i = 1; i != video.size(); ++i)
	{
		//�������ֵ �˴�������ѡ��һ��ѡ��Ϊ��ʹ��opencv�ṩ��calcOpticalFlowFarneback����������
		//һ��ѡ��Ϊʹ��opencv�ṩ��cvCalcOpticalFlowLK��������
		cv::Mat temp;
		//����һ
		{
			cv::calcOpticalFlowFarneback(video[i-1], video[i], temp, 0.5, 3, 15, 3, 5, 1.2, 0);

			//��ģֵ
			cv::Mat magnitude(temp.size(), CV_32FC1);
			for (int row = 0; row != temp.rows; ++row)
			{
				float* dataPtr = magnitude.ptr<float>(row);
#pragma omp parallel for
				for (int col = 0; col < temp.cols; ++col)
				{
					//��������Ĺ���ֵ
					float data1 = temp.at<cv::Vec2f>(row, col)[0];
					float data2 = temp.at<cv::Vec2f>(row, col)[1];

					dataPtr[col] = std::sqrt(std::powf(data1, 2.0f) + std::powf(data2, 2.0f));
				}
			}
			optFlowMagnitude.push_back(magnitude);
		}
		
		//������
//		{
//			cv::Mat mVelx;
//			cv::Mat mVely;
//			if (i - 1 == 0)
//			{
//				velx = cvCreateImage(cvSize(video[i].cols, video[i].rows), IPL_DEPTH_32F, 1);
//				vely = cvCreateImage(cvSize(video[i].cols, video[i].rows), IPL_DEPTH_32F, 1);
//				pCurrGray = cvCreateImage(cvSize(video[i].cols, video[i].rows), IPL_DEPTH_8U, 1);
//				pPreGray = cvCreateImage(cvSize(video[i].cols, video[i].rows), IPL_DEPTH_8U, 1);
//
//				pPreGray->imageData = (char*)video[i - 1].data;
//			}
//
//			pCurrGray->imageData = (char*)video[i].data;
//			cvCalcOpticalFlowLK(pPreGray, pCurrGray, cvSize(5, 5), velx, vely);
//			mVelx = velx;
//			mVely = vely;
//			cvCopy(pCurrGray, pPreGray);
//			
//			//��ģֵ
//			cv::Mat magnitude(mVelx.size(), CV_32FC1);
//			for (int row = 0; row != mVelx.rows; ++row)
//			{
//				float* dataPtr = magnitude.ptr<float>(row);
//#pragma omp parallel for
//				for (int col = 0; col < mVelx.cols; ++col)
//				{
//					//��������Ĺ���ֵ
//					float data1 = mVelx.at<float>(row, col);
//					float data2 = mVely.at<float>(row, col);
//
//					dataPtr[col] = std::sqrt(std::powf(data1, 2.0f) + std::powf(data2, 2.0f));
//				}
//			}
//
//			cv::Mat pyr(cv::Size((magnitude.cols & -2)/2, (magnitude.rows & -2)/2), CV_32FC1);
//			cv::pyrDown(magnitude, pyr);
//			cv::erode(pyr, pyr, cv::Mat(), cv::Point(-1, -1), 1);
//			cv::dilate(pyr, pyr, cv::Mat(), cv::Point(-1, -1), 2);
//			cv::pyrUp(pyr, magnitude);
//
//			//cv::erode(magnitude, magnitude, cv::Mat(), cv::Point(-1, -1), 2);
//			//cv::dilate(magnitude, magnitude, cv::Mat(), cv::Point(-1, -1), 3);
//			cv::blur(magnitude, magnitude, cvSize(5, 5));
//			optFlowMagnitude.push_back(magnitude);
//			//cv::imshow("aaa", magnitude);
//			//cvWaitKey(1);
//		}
	}

	//����ָʾֵ
	std::vector<cv::Mat> indicator(optFlowMagnitude.size(), 
								   cv::Mat::zeros(optFlowMagnitude[0].size(), CV_32F));
	float num = static_cast<float>(optFlowMagnitude[0].cols * optFlowMagnitude[0].rows);
	
	//����һ��Ԫ��
	{
		float aver = static_cast<float>(cv::sum(optFlowMagnitude[0])[0] / num);
		cv::Mat temp = (optFlowMagnitude[0] > aver) / 255;

		//��ֹ�������������������תΪCV_32F
		temp.convertTo(temp, CV_32F);

		indicator[0] = temp;
	}

	const int bound = static_cast<int>(optFlowMagnitude.size());
	
	//������ֵ
#pragma omp parallel for
	for(int i = 1; i < bound; ++i)
	{
		//�����ֵ
		cv::Mat diff;
		cv::absdiff(optFlowMagnitude[i], optFlowMagnitude[i - 1], diff);
		
		//����ƽ��ֵ
		float aver = static_cast<float>(cv::sum(diff)[0] / num);

		//Ϊindicator��ֵ
		cv::Mat temp = (optFlowMagnitude[i] > aver) / 255;

		//��ֹ�������������������תΪCV_32F
		temp.convertTo(temp, CV_32F);

		indicator[i] = temp;
	}

	const float scale = static_cast<float>(1.0f / indicator.size());
	desc = std::accumulate(indicator.begin(), indicator.end(),
		                   cv::Mat::zeros(indicator[0].size(), CV_32FC1)) * scale;
		
}

/*
 *@brief �õ���Ƶ����VIF��������ÿһ�д���һ����Ƶ
 *			��Ƶ�ߴ��Լ���Ƶ������Ƶ�ĸ���ΪӲ���룬�Ժ�Ҫ��
 *param[in] folderName ��Ƶ���ļ�������
 *			videoNum ��Ƶ������Ƶ����
 *			videoSz ��Ƶ�ߴ�
 *param[out] desc ��Ƶ������������
*/
void queryVIFVec(const std::string& folderName, const int videoNum, 
	             const cv::Size2i& videoSz, cv::Mat& desc)
{
	//�����Ϸ��Լ��
	if (videoNum < 1)
	{
		throw("�������Ƶ������Ч--queryVIFVec");
	}

	int videoSize = videoSz.height * videoSz.width;
	desc = cv::Mat::zeros(videoNum, videoSize, CV_32F);
	VIF vif;
	
	for (int i = 1; i <= videoNum; ++i)
	{
		//������Ƶ
		std::stringstream ss;
		ss << folderName << i << ".avi";
			
		cv::VideoCapture capture(ss.str());
		if (!capture.isOpened())
		{
			std::cout << "�޷�����Ƶ�ļ���" << ss.str() << std::endl;
			throw("�޷�����Ƶ�ļ�---queryVIFVec");
			return;
		}
		//��Ƶʵ�ʳߴ�
		const int width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
		const int height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));

		//�Ƿ���Ҫ�ߴ�ת���ı�־
		const bool cvtFlag = (width != videoSz.width) || (height != videoSz.height);

		//��ȡ��Ƶ֡
		cv::Mat img;
		std::vector<cv::Mat> video;		
		if(cvtFlag)
		{
			//ת����Ƶ֡�ߴ��Լ���ɫ������
			while(capture.read(img))
			{
				cv::Mat frame;
				cv::cvtColor(img, frame, CV_BGR2GRAY);
				cv::Mat temp;
				cv::resize(frame, temp, cv::Size2i(videoSz.width, videoSz.height), 
					       0, 0, cv::INTER_LANCZOS4);
				video.push_back(temp);
			}
		}
		else
		{
			//ת����Ƶ��ɫ������
			while(capture.read(img))
			{
				cv::Mat frame;
				cv::cvtColor(img, frame, CV_BGR2GRAY);
				video.push_back(frame);
			}
		}
		capture.release();

		//��ȡ��Ƶ��VIF����
		cv::Mat tempDesc;
		vif.descripVideoG(video, tempDesc);

		//������ת��Ϊ��������Ȼ�󱣴�
		cv::Mat rowDesc;
		cvtMat2Row(tempDesc, rowDesc);
		rowDesc.copyTo(desc.row(i - 1));		
	}
}

/*
 *@brief ��ȡ��Ƶ����VIF������Ĭ���ļ�������NonViolence��violence�����ļ��� 
 *		 ��ͬ����Ƶ�����ܾ��в�ͬ��·����������õ���Ƶ��·���������Ҫ�ر�ע��
 *@param[in]videoSet ��Ƶ��
 *@param[out]nonViolenceVIF nonViolence��Ƶ����VIF������violence��Ƶ��������
*/
void queryVideoSetVIF(const VideoSet& videoSet, cv::Mat& nonViolenceVIF, cv::Mat& violenceVIF)
{
	//���ز�����Ƶ����ȡVIF����
	{
		std::stringstream ss;
		ss << "data/" << videoSet.videosetName << "/NonViolence/";	
		queryVIFVec(ss.str(), videoSet.videoNum, videoSet.videoSz, nonViolenceVIF);

		if (nonViolenceVIF.rows != videoSet.videoNum)
		{
			std::cout << "��ȡ��VIF�����Ƿ�������ߴ��Ԥ��Ĳ�ͬ---queryVideoSetVIF" << std::endl;
			throw("��ȡ��VIF�����Ƿ�������ߴ��Ԥ��Ĳ�ͬ---queryVideoSetVIF");
		}
		std::cout << "nonViolence over" << std::endl;
	}
	
	{
	    std::stringstream ss;
		ss << "data/" << videoSet.videosetName << "/violence/";
		queryVIFVec(ss.str(), videoSet.videoNum, videoSet.videoSz, violenceVIF);
		
		if (violenceVIF.rows != videoSet.videoNum)
		{
			std::cout << "��ȡ��VIF�����Ƿ�---queryVideoSetVIF" << std::endl;
			throw("��ȡ��VIF�����Ƿ�---queryVideoSetVIF");
		}
		std::cout << "violence over" << std::endl;
	}	
}