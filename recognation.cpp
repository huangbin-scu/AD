/*
 *�����ָ�����Ƶ����Ϊ��ʶ��
*/

#include "vif.h"
#include "CrowdDensity.h"

bool detectFlag  = 1;

/*
 *����ѵ���õ��ķ���������������Ƶ���з���
*/
void recog(const std::pair<std::string, cv::Size2i>& clfSz, cv::VideoCapture& capture,
           float &label)
{
	if (0 == clfSz.first.length() || !capture.isOpened())
	{
		std::cout << "�Ƿ�����" << std::endl;
		return;
	}

	//���ط�����
	CvSVM svm;
	svm.load(clfSz.first.c_str());

	//��Ƶ֡��
	const int frameCount = capture.get(CV_CAP_PROP_FRAME_COUNT);

	//��ȡ��Ƶ֡
	cv::Mat img;
	std::vector<cv::Mat> video(frameCount, cv::Mat::zeros(clfSz.second, CV_8UC1));		
	std::vector<cv::Mat> colorVideo(frameCount, cv::Mat::zeros(clfSz.second, CV_32FC3));	

	//�ж��Ƿ���Ҫ�Դ������Ƶ���и�ʽת�� 
	bool cvtFlag = false;

	int width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
	int height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));

	//�����Ƶ�ߴ��Ƿ��ѵ��SVMʱʹ�õ���Ƶ�ߴ���ͬ
	if (0 == width || 0 == height)
	{
		std::cout << "�޷���ȡ��Ƶ" << std::endl;
		throw("�޷���ȡ��Ƶ");
	}
	else if (width * height != svm.get_var_count())
	{
		cvtFlag = true;
	}

	//����Ƶ֡���浽video������ 
	if(!queryFrames(capture, frameCount, cvtFlag, video, colorVideo))
	{
		return;
	}
	
	//����Ƶ����������
	cv::Mat desc;
	VIF vif;
	vif.descripVideoG(video, desc);

	cv::Mat testData;
	cvtMat2Row(desc, testData);

	label = svm.predict(testData);
}

/*
 *@����Ƶ������ʶ��Ϊ��ͳ��ʶ����
 *@param[in] svmFilterName ����������
 *			 videoSet ��Ƶ������
 *			 videoNum ��Ƶ����
 *			 videoSz  ��Ƶ�ߴ�
 *@param[out] result ʶ����
*/
void recogVideoSet(const std::string& svmFilterName, const VideoSet& videoSet, float& result)
{
	/*********�˴��޷����Ƿ�ɹ������˷��������м��飬��Ҫע��************/
	//���ط�������
	CvSVM svm;
	svm.load(svmFilterName.c_str());
	
	//��ȡ��Ƶ����VIF����
	cv::Mat nonViolenceData;
	cv::Mat violenceData;

	//��Ӳ���ϴ��и���Ƶ������������ʱ��ֱ�Ӷ�ȡ
	readDesc(videoSet, violenceData, nonViolenceData);

	std::vector<float>nonViolLabels(nonViolenceData.rows, 0);

	//ʶ��nonViolence��Ƶ��
#pragma omp parallel for	
	for (int i = 0; i < nonViolenceData.rows; ++i)
	{
		nonViolLabels[i] = svm.predict(nonViolenceData.row(i));
	}
	//ͳ�ƽ��
	float recogRate1(0);
	std::for_each(nonViolLabels.begin(), nonViolLabels.end(), [&recogRate1](float& val)
	{
		if(0.0f == val)
		{
			recogRate1 += 1.0f;
		}
	});

	std::vector<float> violLabels(violenceData.rows, 1);

	//ʶ��violence��Ƶ��
#pragma omp parallel for
	for (int i = 0; i < violenceData.rows; ++i)
	{
		violLabels[i] = svm.predict(violenceData.row(i));
	}
	
	//ͳ�ƽ��
	float recogRate2(0);
	std::for_each(violLabels.begin(), violLabels.end(), [&recogRate2](float& val)
	{
		if(1.0f == val)
		{
			recogRate2 += 1.0f;
		}
	});

	std::cout << "���������������쳣ʶ��Ϊ�쳣����" << recogRate2
		      << "\t������������������ʶ��Ϊ�쳣����" << nonViolLabels.size() - recogRate1 << std::endl
		      << "���������������쳣ʶ��Ϊ��������" << violLabels.size() - recogRate2
		      << "\t������������������ʶ��Ϊ��������" << recogRate1 << std::endl;

	result = (recogRate1 + recogRate2) / (nonViolLabels.size() + violLabels.size());
	std::cout << "��ȷ��(ACC)Ϊ��" << result << std::endl;
}

/*
 *���ĳһ����Ƶ����ʶ�����
*/
void recogSingleVideo(const std::pair<std::string, cv::Size2i>& clfSz)
{	
	cv::VideoCapture capture;
	CCapture myCapture(clfSz.second.width, clfSz.second.height);

	//��ʼ����Ƶ��ȡԴ
	if (!myCapture.initCapture(capture))
	{
		capture.release();
		return;
	}
	
	//���÷���������ʶ��
	float label = 0;

	//ͳ�Ƴ����ʱ���
	std::clock_t start = std::clock();

	//����Ƶ����ʶ��
	recog(clfSz, capture, label);
	

	//system("pause");

	capture.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
	
	if (0 == label)
	{
		myCapture.showVideo(capture, "Normal");
	}
	else if (1 == label)
	{
		cout << endl;
		cout << "ɧ�ҷּ�...." << endl;
		singleRankEstimate(capture);
	}
	else
	{
		std::cout << "labelֵ�Ƿ�" << std::endl;
	}

	std::clock_t end = std::clock();
	std::cout << "��ʱ��" << end - start << std::endl
		<< "֡����" << capture.get(CV_CAP_PROP_FRAME_COUNT) << std::endl
		<< "ÿ֡����ʱ�䣺" << (end - start) / capture.get(CV_CAP_PROP_FRAME_COUNT) << std::endl;
	
	capture.release();
}

/*
 *��Գ���Ƶ����ʶ��Ҳ����˵������Ҫÿ������֡����һ������ǰ��֡��Ϊ��ʶ������
 *����Ƶ�������һ����Ƶ��ֻ����һ����Ϊ����Ϊ��������Ƶ��˵��
*/

int gNum = 0;//�Ѿ��������Ƶ����Ŀ
bool gEndFlag = false;//�Ƿ���ʾ���
std::map<int, float> gMeanTime;//ÿ����Ƶ��ƽ������ʱ��
HANDLE hMutexgEndFlag;//���ƶ�gEndFlag�Ļ������
HANDLE hMutexgMeanTime;//���ƶԸ�MeanTime�Ļ������

void recogLongVideo(const std::pair<std::string, cv::Size2i>& clfSz)
{
	cv::VideoCapture capture;

	//��ʼ����Ƶ��ȡԴ
	CCapture myCapture(clfSz.second.width, clfSz.second.height);
	if (!myCapture.initCapture(capture))
	{
		capture.release();
		return;
	}
	float label = -1;

	//std::clock_t start = std::clock();
	//���ط�����
	CvSVM svm;
	svm.load(clfSz.first.c_str());
	
	//�ж��Ƿ���Ҫ�Դ������Ƶ���и�ʽת�� 
	bool cvtFlag = false;

	const int width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
	const int height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));

	//�����Ƶ�ߴ��Ƿ��ѵ��SVMʱʹ�õ���Ƶ�ߴ���ͬ
	if ((width != clfSz.second.width) || (height != clfSz.second.height))
	{
		cvtFlag = true;
	}
	
	//������Ƶ֡��
	const double fps = capture.get(CV_CAP_PROP_FPS);
	const int delay = static_cast<int>(1000 / fps) - 10;
	
	//����Ϊ��Ƶ
	cv::VideoWriter writer("LongVideo.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, cv::Size(320, 240));
	
	std::vector<cv::Mat> video(FRAME_NUM, cv::Mat::zeros(clfSz.second, CV_8UC1));
	std::vector<cv::Mat> colorVideo(FRAME_NUM, cv::Mat::zeros(clfSz.second, CV_32FC3));

	//����������ʾ���߳�
	HANDLE hThread1;
	hThread1 = CreateThread(NULL, 0, Fun1Proc, NULL, 0, NULL);
	CloseHandle(hThread1);

	//�����������
	hMutexgEndFlag = CreateMutex(NULL, true, (LPCWSTR)("gEndFlag"));
	ReleaseMutex(hMutexgEndFlag);
	hMutexgMeanTime = CreateMutex(NULL, true, (LPCWSTR)("gMeanTime"));
	ReleaseMutex(hMutexgMeanTime);

	int vedioNum = 0; //����Ƶ�ε����к�
	while(1)
	{
		//�����ʱͳ��
		std::clock_t start = std::clock();
		std::clock_t end = 0;

		//��ȡ��Ƶ֡
		if (!queryFrames(capture, FRAME_NUM, cvtFlag, video, colorVideo))
		{
			break;
		}

		vedioNum ++;
		gNum = vedioNum;

		//��ȡ��Ƶ��������
		cv::Mat desc;
		VIF vif;
		vif.descripVideoG(video, desc);

		//��������ת��Ϊ��������
		cv::Mat testData;
		cvtMat2Row(desc, testData);

		//ʹ��֧������������ʶ��
		label = svm.predict(testData);

		//��������Ķ���Ƶ��
		stringstream sVedioNum;
		sVedioNum << vedioNum;
		string slipWriterName = "LongVedioSlip" + sVedioNum.str() + ".avi";
		cv::VideoWriter slipWriter(slipWriterName, CV_FOURCC('M', 'J', 'P', 'G'), 25.0, cv::Size(320, 240));

		if (0 == label)
		{
			if (0 == myCapture.showFrames(colorVideo, /*"nonViolence"*/"Normal", /*delay,*/ writer, slipWriter))
			{
				break;
			}

		}
		else if (1 == label)
		{
			//if(0 == myCapture.showFrames(colorVideo, /*"nonViolence"*/"Abnormal", writer, slipWriter))
			//{
			//	break;
			//}
			cout << endl;
			cout << "ɧ�ҷּ�...." << endl;
			cout << endl;
			detectFlag = true;
			longRankEstimate(colorVideo, writer, slipWriter);
		}
		else
		{
			std::cout << "labelֵ�Ƿ�" << std::endl;
		}

		end = std::clock();
		std::cout << "��ʱ��" << end - start << std::endl
			      << "ÿ֡��ʱ��" << (end - start) / static_cast<float>(FRAME_NUM) << std::endl;

		float meanTime = (end - start) / static_cast<float>(FRAME_NUM);

		//����ÿ����Ƶ�ε�ƽ������ʱ��
		WaitForSingleObject(hMutexgMeanTime, INFINITE);
 		gMeanTime.insert(std::pair<int, float>(vedioNum, meanTime));
		ReleaseMutex(hMutexgMeanTime);
	}

	//�����̴߳�����󣬵ȴ���ʾ�̴߳������
	while (true)
	{
		WaitForSingleObject(hMutexgEndFlag, INFINITE);
		if (gEndFlag)
			break;
		ReleaseMutex(hMutexgEndFlag);
	}

	capture.release();
}

DWORD WINAPI Fun1Proc(LPVOID lpParameter)//thread data
{
	//�ж��Ƿ���һ���Ѿ�����õĶ���Ƶ�Σ����û�У���һֱ��ѯ�ȴ�
	while(1)
	{
		if (gNum > 1)
			break;

		Sleep(1000);
	}

	int loopnum = 1;
	while (true)
	{
		int waitCount = 0;
		//���Ҫ�������Ƶ��δ�����꣬��ȴ�
		while (gNum < loopnum)
		{
			waitCount ++;
			if (waitCount == 5)
			{
				break;
			}

			Sleep(1000);
		}

		//����ȴ�waitCount=5��δ�����꣬����ж���������Ƶ�Ѿ��������
		if (gNum < loopnum)
		{
			cout << "vedio over!" << endl;
			WaitForSingleObject(hMutexgEndFlag, INFINITE);
			//gEndFlag = true;
			ReleaseMutex(hMutexgEndFlag);
			break;
		}

		cv::Mat frame; 
		std::stringstream ss1;
		ss1 << loopnum;
		string str = "LongVedioSlip" + ss1.str() + ".avi";

		cv::VideoCapture cap(str);
		if (!cap.isOpened())
		{
			cout << "can't open the file" << str << endl;
			break;
		}

		//���뵱ǰҪ��ʾ����Ƶ�ε�ƽ������ʱ��
		WaitForSingleObject(hMutexgMeanTime, INFINITE);
		map<int, float>::iterator meanTimeIter = gMeanTime.find(loopnum);
		float meanTime = meanTimeIter->second;
		gMeanTime.erase(loopnum);
		ReleaseMutex(hMutexgMeanTime);
		
		while (cap.read(frame))
		{	
			imshow("test",frame);
			if (meanTime < 40)
				cv::waitKey(40);
			else
				cv::waitKey(meanTime);
		}
		cap.release();

		//ÿ����һ����Ƶ����ɾ����
		remove(str.c_str());
		loopnum++;
	}
	return 0;
}

/*
 *���ĳһ����Ϊ�ָ�õĶ���Ƶ������Ϊ�쳣�ģ�������쳣�ȼ����з���
*/
void singleRankEstimate(cv::VideoCapture& capture)
{
	CCrowdDensity* pdensity = new CCrowdDensity();
	int frameNum = 0; 
	
	cv::Mat imgMat;
	capture.read(imgMat);
	IplImage *pframe;
	pframe = cvCreateImage(cvSize(imgMat.cols, imgMat.rows), 8, 3); 
	pframe->imageData = (char*)imgMat.data;

	pdensity->InitialCrowdDensity(pframe);
	pdensity->VedioDetect(pframe, frameNum);
	frameNum++;

	const double fps = capture.get(CV_CAP_PROP_FPS);
	const int delay = static_cast<int>(1000 / fps) - 20;
	cv::VideoWriter writer("SingleVideo.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, cv::Size(320, 240));

	char* m_text;
	CvFont font; //ͼƬ���������
	cvInitFont(&font, CV_FONT_HERSHEY_TRIPLEX, 0.5, 0.5, 0, 1, CV_AA); //��ʼ������

	while (capture.read(imgMat))
	{
		pframe->imageData =(char*)imgMat.data;

		m_text = pdensity->DesityRank(pframe, frameNum);

		if (!detectFlag)
		{
			cv::Mat img = pframe;
			cv::Mat dst;

			const bool writeTextFlag = (sizeof(m_text) != 0);
			if(writeTextFlag)
			{
				//��ʾ��Ⱥ״̬
				char* CrowdState = "Abnormal";
				CvSize text_size;
				cvGetTextSize(m_text, &font, &text_size, NULL);
				cvPutText(pframe, CrowdState, cvPoint(3, text_size.height + 5), &font, CV_RGB(255, 0, 255));

				//��ʾ�ܶȵȼ�
				cvPutText(pframe, m_text, cvPoint(3, text_size.height + 25), &font, CV_RGB(255, 0, 0));
			}

			cv::resize(img, dst, cv::Size2i(SHOW_W, SHOW_H), 0, 0, cv::INTER_LANCZOS4);
			cv::imshow("img", dst);

			writer << dst;
			if(27 == cv::waitKey(/*delay*/1))
			{
				cv::destroyAllWindows();
				return;
			}	
		}

		if (frameNum == JUDGE_FRAME_DETECT - 1 && detectFlag)
		{
			capture.set(CV_CAP_PROP_POS_FRAMES, 0);
			detectFlag = 0;
			frameNum = 0;
		}
		
		frameNum ++;
	}

	cvDestroyAllWindows();
	cvReleaseImage(&pframe);
	delete pdensity;
}

/*
 *���ĳһ����Ƶ��ÿ��һ��ʱ�����һ���Ƿ�Ϊ�쳣��Ϊ��ʶ������
 *����Ϊ�쳣�ģ�������쳣�ȼ����з���
*/
void longRankEstimate(std::vector<cv::Mat>& videoSeq, cv::VideoWriter& writer, cv::VideoWriter& slipWriter)
{
	CCrowdDensity* pdensity = new CCrowdDensity();
	int frameNum = 0; 

	cv::Mat imgMat;
	imgMat = videoSeq[0];
	IplImage *pframe;
	pframe = cvCreateImage(cvSize(imgMat.cols, imgMat.rows), 8, 3); 
	pframe->imageData = (char*)imgMat.data;

	pdensity->InitialCrowdDensity(pframe);
	pdensity->VedioDetect(pframe, frameNum);
	frameNum++;

	char* m_text;
	CvFont font; //ͼƬ���������
	cvInitFont(&font, CV_FONT_HERSHEY_TRIPLEX, 0.5, 0.5, 0, 1, CV_AA); //��ʼ������

	for (int i = 1; i < videoSeq.size(); i++)
	{
		imgMat = videoSeq[i];
		pframe->imageData = (char*)imgMat.data;

		m_text = pdensity->DesityRank(pframe, frameNum);

		if (!detectFlag)
		{
			cv::Mat img = pframe;
			cv::Mat dst;

			const bool writeTextFlag = (sizeof(m_text) != 0);
			if(writeTextFlag)
			{
				//��ʾ��Ⱥ״̬
				char* CrowdState = "Abnormal";
				CvSize text_size;
				cvGetTextSize(m_text, &font, &text_size, NULL);
				cvPutText(pframe, CrowdState, cvPoint(3, text_size.height + 5), &font, CV_RGB(255, 0, 255));

				//��ʾ�ܶȵȼ�
				cvPutText(pframe, m_text, cvPoint(3, text_size.height + 25), &font, CV_RGB(255, 0, 0));
			}

			cv::resize(img, dst, cv::Size2i(SHOW_W, SHOW_H), 0, 0, cv::INTER_LANCZOS4);
			//cv::imshow("img", dst);

			writer << dst;
			slipWriter << dst;
			//if(27 == cv::waitKey(1))
			//{
			//	cv::destroyAllWindows();
			//	return;
			//}	
		}

		if (frameNum == JUDGE_FRAME_DETECT - 1 && detectFlag)
		{
			i = 0;
			detectFlag = 0;
			frameNum = 0;
		}
		frameNum++;
	}

	cvDestroyWindow("img");
	cvReleaseImage(&pframe);
	delete pdensity;
}