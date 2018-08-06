#include "Capture.h"

// �����ѵ�����Ĺ��캯����
// �й��ඨ�����Ϣ������� Capture.h
CCapture::CCapture(const int width = 320, const int height = 240)
	               :m_width(width), m_height(height)
{
	return;
}

/*
 *@brief ��ȡ��Ƶ������ͷ����
*/
bool CCapture::initCapture(cv::VideoCapture& capture)
{
	if (capture.isOpened())
	{
		return false;
	}

	//ģʽѡ�񲿷�
	std::cout << "��ѡ��ģʽ:" << std::endl
		      << "\t1---������Ƶ���ݣ�2--��������ͷʵʱ����" << std::endl;
	int mode = -1;
	std::cin >> mode;
	std::string fileName("");
	if (1 == mode)
	{
		std::cout << "��ѡ������Ƶ���ݣ���������Ƶ���ƣ���1.avi" << std::endl;
		std::cin >> fileName;

		//����Ƶ�ļ� 
		if (!capture.open(fileName))
		{
			std::cout << "�޷�����Ƶ�ļ���" << fileName << std::endl;
			throw("�޷�����Ƶ�ļ�");
			return false;
		}
	}
	else if (2 == mode)
	{
		//������ͷ 
		if (!capture.open(0))     
		{   
			fprintf(stderr, "�޷�������ͷ!\n");   
			system("pause");
			return false;     
		}
		capture.set(CV_CAP_PROP_FRAME_WIDTH, m_width);
		capture.set(CV_CAP_PROP_FRAME_HEIGHT, m_height);
	}
	else
	{
		throw("�Ƿ����룬ֻ������1����2");
	}
	return true;
}

/*
 *@brief ��ʾͼ������
 *@param[in]frames ͼ������
 *			text Ҫ��ͼ����д����
 *			delay ͼ����ʾʱ����ʱʱ��
*/
int CCapture::showFrames(std::vector<cv::Mat>& frames, const std::string& text,/* const int delay,*/
	                     cv::VideoWriter& writer,  cv::VideoWriter& slipWriter)
{
	if(frames.size() == 0)
	{
		return 0;
	}

	//���гߴ�ת����־
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

		//����Ϊ��Ƶ
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

//��ʾ��Ƶ������Ƶ��д�϶�Ӧ������
/*
 *@brief ��ʾ��Ƶ������Ƶ��д�϶�Ӧ������
 *@param[in]videoName ��Ƶ�� textҪд����Ƶ�ϵ�����
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

	//��������
	if (27 == cv::waitKey(0))
	{
		cv::destroyAllWindows();
		return;
	}
}