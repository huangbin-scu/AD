/*
 *该部分负责对视频采用光流幅值变化量进行描述
*/

#include "vif.h"
#include <numeric>
#include <ctime>

/*
 *@brief 描述视频  globle descriptor全局描述，使用vif向量作为视频的全局描述
 *@param[in]video 视频数据
 *@param[out]desc 得到的描述向量
*/
void VIF::descripVideoG(const std::vector<cv::Mat>& video, cv::Mat& desc)
{
	//求取光流模值
	std::vector<cv::Mat> optFlowMagnitude;

	IplImage* velx, *vely, *pCurrGray, *pPreGray;
	
	for (int i = 1; i != video.size(); ++i)
	{
		//计算光流值 此处有两个选择一种选择为是使用opencv提供的calcOpticalFlowFarneback光流函数，
		//一种选择为使用opencv提供的cvCalcOpticalFlowLK光流函数
		cv::Mat temp;
		//方法一
		{
			cv::calcOpticalFlowFarneback(video[i-1], video[i], temp, 0.5, 3, 15, 3, 5, 1.2, 0);

			//求模值
			cv::Mat magnitude(temp.size(), CV_32FC1);
			for (int row = 0; row != temp.rows; ++row)
			{
				float* dataPtr = magnitude.ptr<float>(row);
#pragma omp parallel for
				for (int col = 0; col < temp.cols; ++col)
				{
					//两个方向的光流值
					float data1 = temp.at<cv::Vec2f>(row, col)[0];
					float data2 = temp.at<cv::Vec2f>(row, col)[1];

					dataPtr[col] = std::sqrt(std::powf(data1, 2.0f) + std::powf(data2, 2.0f));
				}
			}
			optFlowMagnitude.push_back(magnitude);
		}
		
		//方法二
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
//			//求模值
//			cv::Mat magnitude(mVelx.size(), CV_32FC1);
//			for (int row = 0; row != mVelx.rows; ++row)
//			{
//				float* dataPtr = magnitude.ptr<float>(row);
//#pragma omp parallel for
//				for (int col = 0; col < mVelx.cols; ++col)
//				{
//					//两个方向的光流值
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

	//计算指示值
	std::vector<cv::Mat> indicator(optFlowMagnitude.size(), 
								   cv::Mat::zeros(optFlowMagnitude[0].size(), CV_32F));
	float num = static_cast<float>(optFlowMagnitude[0].cols * optFlowMagnitude[0].rows);
	
	//填充第一个元素
	{
		float aver = static_cast<float>(cv::sum(optFlowMagnitude[0])[0] / num);
		cv::Mat temp = (optFlowMagnitude[0] > aver) / 255;

		//防止数据溢出，将数据类型转为CV_32F
		temp.convertTo(temp, CV_32F);

		indicator[0] = temp;
	}

	const int bound = static_cast<int>(optFlowMagnitude.size());
	
	//填充后续值
#pragma omp parallel for
	for(int i = 1; i < bound; ++i)
	{
		//计算差值
		cv::Mat diff;
		cv::absdiff(optFlowMagnitude[i], optFlowMagnitude[i - 1], diff);
		
		//计算平均值
		float aver = static_cast<float>(cv::sum(diff)[0] / num);

		//为indicator赋值
		cv::Mat temp = (optFlowMagnitude[i] > aver) / 255;

		//防止数据溢出，将数据类型转为CV_32F
		temp.convertTo(temp, CV_32F);

		indicator[i] = temp;
	}

	const float scale = static_cast<float>(1.0f / indicator.size());
	desc = std::accumulate(indicator.begin(), indicator.end(),
		                   cv::Mat::zeros(indicator[0].size(), CV_32FC1)) * scale;
		
}

/*
 *@brief 得到视频集的VIF描述矩阵，每一行代表一个视频
 *			视频尺寸以及视频集中视频的个数为硬编码，以后要改
 *param[in] folderName 视频集文件夹名称
 *			videoNum 视频集中视频个数
 *			videoSz 视频尺寸
 *param[out] desc 视频集的描述矩阵
*/
void queryVIFVec(const std::string& folderName, const int videoNum, 
	             const cv::Size2i& videoSz, cv::Mat& desc)
{
	//参数合法性检查
	if (videoNum < 1)
	{
		throw("输入的视频个数无效--queryVIFVec");
	}

	int videoSize = videoSz.height * videoSz.width;
	desc = cv::Mat::zeros(videoNum, videoSize, CV_32F);
	VIF vif;
	
	for (int i = 1; i <= videoNum; ++i)
	{
		//加载视频
		std::stringstream ss;
		ss << folderName << i << ".avi";
			
		cv::VideoCapture capture(ss.str());
		if (!capture.isOpened())
		{
			std::cout << "无法打开视频文件：" << ss.str() << std::endl;
			throw("无法打开视频文件---queryVIFVec");
			return;
		}
		//视频实际尺寸
		const int width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
		const int height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));

		//是否需要尺寸转换的标志
		const bool cvtFlag = (width != videoSz.width) || (height != videoSz.height);

		//获取视频帧
		cv::Mat img;
		std::vector<cv::Mat> video;		
		if(cvtFlag)
		{
			//转换视频帧尺寸以及颜色，保存
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
			//转换视频颜色，保存
			while(capture.read(img))
			{
				cv::Mat frame;
				cv::cvtColor(img, frame, CV_BGR2GRAY);
				video.push_back(frame);
			}
		}
		capture.release();

		//获取视频的VIF描述
		cv::Mat tempDesc;
		vif.descripVideoG(video, tempDesc);

		//将描述转换为行向量，然后保存
		cv::Mat rowDesc;
		cvtMat2Row(tempDesc, rowDesc);
		rowDesc.copyTo(desc.row(i - 1));		
	}
}

/*
 *@brief 提取视频集的VIF描述，默认文件夹内有NonViolence和violence两个文件夹 
 *		 不同的视频集可能具有不同的路径，故下面得到视频集路径的语句需要特别注意
 *@param[in]videoSet 视频集
 *@param[out]nonViolenceVIF nonViolence视频集的VIF描述，violence视频集的描述
*/
void queryVideoSetVIF(const VideoSet& videoSet, cv::Mat& nonViolenceVIF, cv::Mat& violenceVIF)
{
	//加载测试视频，提取VIF描述
	{
		std::stringstream ss;
		ss << "data/" << videoSet.videosetName << "/NonViolence/";	
		queryVIFVec(ss.str(), videoSet.videoNum, videoSet.videoSz, nonViolenceVIF);

		if (nonViolenceVIF.rows != videoSet.videoNum)
		{
			std::cout << "提取的VIF描述非法，结果尺寸跟预想的不同---queryVideoSetVIF" << std::endl;
			throw("提取的VIF描述非法，结果尺寸跟预想的不同---queryVideoSetVIF");
		}
		std::cout << "nonViolence over" << std::endl;
	}
	
	{
	    std::stringstream ss;
		ss << "data/" << videoSet.videosetName << "/violence/";
		queryVIFVec(ss.str(), videoSet.videoNum, videoSet.videoSz, violenceVIF);
		
		if (violenceVIF.rows != videoSet.videoNum)
		{
			std::cout << "提取的VIF描述非法---queryVideoSetVIF" << std::endl;
			throw("提取的VIF描述非法---queryVideoSetVIF");
		}
		std::cout << "violence over" << std::endl;
	}	
}