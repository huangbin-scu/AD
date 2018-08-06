#include "Capture.h"

// 这是已导出类的构造函数。
// 有关类定义的信息，请参阅 Capture.h
CCapture::CCapture(const int width = 320, const int height = 240)
	               :m_width(width), m_height(height)
{
	return;
}

/*
 *@brief 获取视频或摄像头数据
*/
bool CCapture::initCapture(cv::VideoCapture& capture)
{
	if (capture.isOpened())
	{
		return false;
	}

	//模式选择部分
	std::cout << "请选择模式:" << std::endl
		      << "\t1---处理视频数据；2--处理摄像头实时数据" << std::endl;
	int mode = -1;
	std::cin >> mode;
	std::string fileName("");
	if (1 == mode)
	{
		std::cout << "您选择处理视频数据，请输入视频名称：如1.avi" << std::endl;
		std::cin >> fileName;

		//打开视频文件 
		if (!capture.open(fileName))
		{
			std::cout << "无法打开视频文件：" << fileName << std::endl;
			throw("无法打开视频文件");
			return false;
		}
	}
	else if (2 == mode)
	{
		//打开摄像头 
		if (!capture.open(0))     
		{   
			fprintf(stderr, "无法打开摄像头!\n");   
			system("pause");
			return false;     
		}
		capture.set(CV_CAP_PROP_FRAME_WIDTH, m_width);
		capture.set(CV_CAP_PROP_FRAME_HEIGHT, m_height);
	}
	else
	{
		throw("非法输入，只能输入1或者2");
	}
	return true;
}

/*
 *@brief 显示图像序列
 *@param[in]frames 图像序列
 *			text 要在图像上写的字
 *			delay 图像显示时的延时时间
*/
int CCapture::showFrames(std::vector<cv::Mat>& frames, const std::string& text,/* const int delay,*/
	                     cv::VideoWriter& writer,  cv::VideoWriter& slipWriter)
{
	if(frames.size() == 0)
	{
		return 0;
	}

	//进行尺寸转换标志
	bool cvtFlag = (frames[0].rows != SHOW_W)||(frames[0].cols != SHOW_H);

	int len = static_cast<int>(frames.size());
	cv::Mat img;
	for(int i = 0; i != len; ++i)
	{
		if(cvtFlag)
		{			
			cv::resize(frames[i], img, cv::Size2i(SHOW_W, SHOW_H), 0, 0, 
                cv::INTER_LANCZOS4);
		}
		else
		{
			img = frames[i];
		}
		if (text == "Normal")
		cv::putText(img, text, cv::Point(3, static_cast<int>(img.rows * 0.1)),
					cv::FONT_HERSHEY_TRIPLEX, 0.7, cv::Scalar(255, 0, 0, 0), 2, 8);
		if (text == "Abnormal")
			cv::putText(img, text, cv::Point(3, static_cast<int>(img.rows * 0.1)),
			cv::FONT_HERSHEY_TRIPLEX, 0.7, cv::Scalar(0, 0, 255, 0), 2, 8);
		//cv::imshow("img", img);

		//保存为视频
		writer << img;
		slipWriter << img;

		//if(27 == cv::waitKey(delay))
		//{
		//	cv::destroyAllWindows();
		//	return 0;
		//}
		//std::stringstream ss;
		//ss<< i<< ".jpg";
		//cv::imwrite(ss.str(), img);
	}
	return 1;
}

//显示视频，在视频上写上对应的文字
/*
 *@brief 显示视频，在视频上写上对应的文字
 *@param[in]videoName 视频名 text要写在视频上的文字
*/
void CCapture::showVideo(cv::VideoCapture& capture, const std::string& text)
{
	if (!capture.isOpened())
	{		
		std::cout << "can't open the video!" << std::endl;
		throw("can't open the video!");
		return;
	}
	const bool writeTextFlag = (text.length() != 0);

	const double fps = capture.get(CV_CAP_PROP_FPS);
	const int delay = static_cast<int>(1000 / fps) - 10;

	cv::VideoWriter writer("SingleVideo.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0,
		                    cv::Size(320, 240));

	cv::Mat img;
	cv::Mat dst;
	while (capture.read(img))
	{
		if (writeTextFlag)
		{
			cv::putText(img, text, cv::Point(3, static_cast<int>(img.rows * 0.1)),
				        cv::FONT_HERSHEY_TRIPLEX, 0.7, cv::Scalar(255, 0, 0, 0), 2, 8);
		}

		cv::resize(img, dst, cv::Size2i(SHOW_W, SHOW_H), 0, 0, cv::INTER_LANCZOS4);
		cv::imshow("img", dst);
		
		writer << dst;
		if (27 == cv::waitKey(delay))
		{
			cv::destroyAllWindows();
			return;
		}
	}

	//结束程序
	if (27 == cv::waitKey(0))
	{
		cv::destroyAllWindows();
		return;
	}
}