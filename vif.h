/*
 *@brief 使用ViF(violence flow)对行为进行描述 算法详见《Violent Flows Real-Time Detection of Violent Crowd Behavior》
*/
#ifndef VIF_H
#define VIF_H

#include <iostream>
#include <vector>
#include "Capture.h"
#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <cxcore.h>
#include <windows.h>
#include <map>

#ifndef FRAME_NUM
#define FRAME_NUM 50
#endif

//使用ViF描述视频
class VIF
{
public:
	VIF();
	
	//全局描述
	void descripVideoG(const std::vector<cv::Mat>& video, cv::Mat& desc);

	//局部描述得到特征集Bag of Features
	//void descripVideoBoF(const std::vector<cv::Mat>& video, cv::Mat& desc);
private:

};

/*
 *保存视频集信息
*/
//视频集信息
struct VideoSet
{
	int videoNum; //视频个数
	cv::Size2i videoSz; //视频尺寸
	std::string videosetName; //视频集名称

	//构造函数
	VideoSet(const int& num, const cv::Size2i& size, const std::string& name)
		     :videoNum(num), videoSz(size), videosetName(name)
	{}
};

//针对视频集进行训练，视频集中视频命名要按照规定来
void train(const VideoSet& videoSet);

//训练密度统计需要的分类器
void trainDensity();

//对单一的视频进行识别，方便演示
void recog(const std::pair<std::string, cv::Size2i>& clfSz, cv::VideoCapture& capture, float &label);

//对视频集进行识别，为了统计识别率
void recogVideoSet(const std::string& svmFilterName, const VideoSet& videoSet, float& result);

//将视频帧转换为灰度图像保存到mat数组中
bool queryFrames(cv::VideoCapture& capture, const int frameNum,
				 const bool cvtFlag, std::vector<cv::Mat>& frames, std::vector<cv::Mat>& colorFrames);

//针对视频集计算描述矩阵
void queryVIFVec(const std::string& folderName, 
                 const int videoNum, const cv::Size2i& videoSz, cv::Mat& desc);

//提取视频集的VIF描述
void queryVideoSetVIF(const VideoSet& videoSet, cv::Mat& nonViolenceVIF, cv::Mat& violenceVIF);

//将矩阵转换为行向量
void cvtMat2Row(const cv::Mat& input, cv::Mat& output);

//针对某一个分割好的短视频进行识别测试
void recogSingleVideo(const std::pair<std::string, cv::Size2i>& clfSz);

//针对某一个长视频进行识别测试
void recogLongVideo(const std::pair<std::string, cv::Size2i>& clfSz);

//针对5个人群骚乱视频的交叉测试
float crossValidation();

//尝试从硬盘读取视频集的描述，如果读取失败，使用算法计算，同时将结果保存到硬盘上
void readDesc(const VideoSet& videoSet, cv::Mat& violenceData, cv::Mat& nonViolenceData);

//针对某一个分割好的异常短视频进行分级
void singleRankEstimate(cv::VideoCapture& capture);

//针对某一个异常长视频进行分级
void longRankEstimate(std::vector<cv::Mat>& videoSeq, cv::VideoWriter& writer, cv::VideoWriter& slipWriter);

DWORD WINAPI Fun1Proc(LPVOID lpParameter);

#endif