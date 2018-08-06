
//��Ƶ��ʾ���
#ifndef SHOW_W
#define SHOW_W 320
#endif

//��Ƶ��ʾ�߶�
#ifndef SHOW_H
#define SHOW_H 240
#endif

#include <opencv2\opencv.hpp>
#include <string>

// �����Ǵ� Capture.dll ������
class CCapture {
public:
	CCapture(const int width, const int height); 

	//��ʾ��Ƶ������Ƶ��д�϶�Ӧ������
	void showVideo(cv::VideoCapture& capture, const std::string& text);

	//��ʾͼƬ����
	int showFrames(std::vector<cv::Mat>& frames, const std::string& text, /*const int delay,*/ 
		           cv::VideoWriter& writer, cv::VideoWriter& slipWriter);

	//��ȡ��Ƶ������ͷ����
	bool initCapture(cv::VideoCapture& capture);

private:
	int m_width;//��Ƶ�Ŀ��
	int m_height;//��Ƶ�ĸ߶�
};
