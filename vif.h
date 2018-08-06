/*
 *@brief ʹ��ViF(violence flow)����Ϊ�������� �㷨�����Violent Flows Real-Time Detection of Violent Crowd Behavior��
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

//ʹ��ViF������Ƶ
class VIF
{
public:
	VIF();
	
	//ȫ������
	void descripVideoG(const std::vector<cv::Mat>& video, cv::Mat& desc);

	//�ֲ������õ�������Bag of Features
	//void descripVideoBoF(const std::vector<cv::Mat>& video, cv::Mat& desc);
private:

};

/*
 *������Ƶ����Ϣ
*/
//��Ƶ����Ϣ
struct VideoSet
{
	int videoNum; //��Ƶ����
	cv::Size2i videoSz; //��Ƶ�ߴ�
	std::string videosetName; //��Ƶ������

	//���캯��
	VideoSet(const int& num, const cv::Size2i& size, const std::string& name)
		     :videoNum(num), videoSz(size), videosetName(name)
	{}
};

//�����Ƶ������ѵ������Ƶ������Ƶ����Ҫ���չ涨��
void train(const VideoSet& videoSet);

//ѵ���ܶ�ͳ����Ҫ�ķ�����
void trainDensity();

//�Ե�һ����Ƶ����ʶ�𣬷�����ʾ
void recog(const std::pair<std::string, cv::Size2i>& clfSz, cv::VideoCapture& capture, float &label);

//����Ƶ������ʶ��Ϊ��ͳ��ʶ����
void recogVideoSet(const std::string& svmFilterName, const VideoSet& videoSet, float& result);

//����Ƶ֡ת��Ϊ�Ҷ�ͼ�񱣴浽mat������
bool queryFrames(cv::VideoCapture& capture, const int frameNum,
				 const bool cvtFlag, std::vector<cv::Mat>& frames, std::vector<cv::Mat>& colorFrames);

//�����Ƶ��������������
void queryVIFVec(const std::string& folderName, 
                 const int videoNum, const cv::Size2i& videoSz, cv::Mat& desc);

//��ȡ��Ƶ����VIF����
void queryVideoSetVIF(const VideoSet& videoSet, cv::Mat& nonViolenceVIF, cv::Mat& violenceVIF);

//������ת��Ϊ������
void cvtMat2Row(const cv::Mat& input, cv::Mat& output);

//���ĳһ���ָ�õĶ���Ƶ����ʶ�����
void recogSingleVideo(const std::pair<std::string, cv::Size2i>& clfSz);

//���ĳһ������Ƶ����ʶ�����
void recogLongVideo(const std::pair<std::string, cv::Size2i>& clfSz);

//���5����Ⱥɧ����Ƶ�Ľ������
float crossValidation();

//���Դ�Ӳ�̶�ȡ��Ƶ���������������ȡʧ�ܣ�ʹ���㷨���㣬ͬʱ��������浽Ӳ����
void readDesc(const VideoSet& videoSet, cv::Mat& violenceData, cv::Mat& nonViolenceData);

//���ĳһ���ָ�õ��쳣����Ƶ���зּ�
void singleRankEstimate(cv::VideoCapture& capture);

//���ĳһ���쳣����Ƶ���зּ�
void longRankEstimate(std::vector<cv::Mat>& videoSeq, cv::VideoWriter& writer, cv::VideoWriter& slipWriter);

DWORD WINAPI Fun1Proc(LPVOID lpParameter);

#endif