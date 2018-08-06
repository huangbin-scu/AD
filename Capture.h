
//视频显示宽度
#ifndef SHOW_W
#define SHOW_W 320
#endif

//视频显示高度
#ifndef SHOW_H
#define SHOW_H 240
#endif

#include <opencv2\opencv.hpp>
#include <string>

// 此类是从 Capture.dll 导出的
class CCapture {
public:
	CCapture(const int width, const int height); 

	//显示视频，在视频上写上对应的文字
	void showVideo(cv::VideoCapture& capture, const std::string& text);

	//显示图片序列
	int showFrames(std::vector<cv::Mat>& frames, const std::string& text, /*const int delay,*/ 
		           cv::VideoWriter& writer, cv::VideoWriter& slipWriter);

	//获取视频或摄像头数据
	bool initCapture(cv::VideoCapture& capture);

private:
	int m_width;//视频的宽度
	int m_height;//视频的高度
};
