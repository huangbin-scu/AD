/*
 *本部分负责视频中行为的识别
*/

#include "vif.h"
#include "CrowdDensity.h"

bool detectFlag  = 1;

/*
 *加载训练得到的分类器，对输入视频进行分类
*/
void recog(const std::pair<std::string, cv::Size2i>& clfSz, cv::VideoCapture& capture,
           float &label)
{
	if (0 == clfSz.first.length() || !capture.isOpened())
	{
		std::cout << "非法参数" << std::endl;
		return;
	}

	//加载分类器
	CvSVM svm;
	svm.load(clfSz.first.c_str());

	//视频帧数
	const int frameCount = capture.get(CV_CAP_PROP_FRAME_COUNT);

	//获取视频帧
	cv::Mat img;
	std::vector<cv::Mat> video(frameCount, cv::Mat::zeros(clfSz.second, CV_8UC1));		
	std::vector<cv::Mat> colorVideo(frameCount, cv::Mat::zeros(clfSz.second, CV_32FC3));	

	//判断是否需要对传入的视频进行格式转换 
	bool cvtFlag = false;

	int width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
	int height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));

	//检测视频尺寸是否跟训练SVM时使用的视频尺寸相同
	if (0 == width || 0 == height)
	{
		std::cout << "无法读取视频" << std::endl;
		throw("无法读取视频");
	}
	else if (width * height != svm.get_var_count())
	{
		cvtFlag = true;
	}

	//将视频帧保存到video数组中 
	if(!queryFrames(capture, frameCount, cvtFlag, video, colorVideo))
	{
		return;
	}
	
	//求视频的描述矩阵
	cv::Mat desc;
	VIF vif;
	vif.descripVideoG(video, desc);

	cv::Mat testData;
	cvtMat2Row(desc, testData);

	label = svm.predict(testData);
}

/*
 *@对视频集进行识别，为了统计识别率
 *@param[in] svmFilterName 分类器名称
 *			 videoSet 视频集名称
 *			 videoNum 视频个数
 *			 videoSz  视频尺寸
 *@param[out] result 识别率
*/
void recogVideoSet(const std::string& svmFilterName, const VideoSet& videoSet, float& result)
{
	/*********此处无法对是否成功加载了分类器进行检验，需要注意************/
	//加载分类器，
	CvSVM svm;
	svm.load(svmFilterName.c_str());
	
	//提取视频集的VIF描述
	cv::Mat nonViolenceData;
	cv::Mat violenceData;

	//当硬盘上存有该视频集的描述矩阵时，直接读取
	readDesc(videoSet, violenceData, nonViolenceData);

	std::vector<float>nonViolLabels(nonViolenceData.rows, 0);

	//识别nonViolence视频集
#pragma omp parallel for	
	for (int i = 0; i < nonViolenceData.rows; ++i)
	{
		nonViolLabels[i] = svm.predict(nonViolenceData.row(i));
	}
	//统计结果
	float recogRate1(0);
	std::for_each(nonViolLabels.begin(), nonViolLabels.end(), [&recogRate1](float& val)
	{
		if(0.0f == val)
		{
			recogRate1 += 1.0f;
		}
	});

	std::vector<float> violLabels(violenceData.rows, 1);

	//识别violence视频集
#pragma omp parallel for
	for (int i = 0; i < violenceData.rows; ++i)
	{
		violLabels[i] = svm.predict(violenceData.row(i));
	}
	
	//统计结果
	float recogRate2(0);
	std::for_each(violLabels.begin(), violLabels.end(), [&recogRate2](float& val)
	{
		if(1.0f == val)
		{
			recogRate2 += 1.0f;
		}
	});

	std::cout << "真阳性样本数（异常识别为异常）：" << recogRate2
		      << "\t假阳性样本数（正常识别为异常）：" << nonViolLabels.size() - recogRate1 << std::endl
		      << "假阴性样本数（异常识别为正常）：" << violLabels.size() - recogRate2
		      << "\t真阴性样本数（正常识别为正常）：" << recogRate1 << std::endl;

	result = (recogRate1 + recogRate2) / (nonViolLabels.size() + violLabels.size());
	std::cout << "精确率(ACC)为：" << result << std::endl;
}

/*
 *针对某一个视频进行识别测试
*/
void recogSingleVideo(const std::pair<std::string, cv::Size2i>& clfSz)
{	
	cv::VideoCapture capture;
	CCapture myCapture(clfSz.second.width, clfSz.second.height);

	//初始化视频获取源
	if (!myCapture.initCapture(capture))
	{
		capture.release();
		return;
	}
	
	//调用分类器进行识别
	float label = 0;

	//统计程序耗时情况
	std::clock_t start = std::clock();

	//对视频进行识别
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
		cout << "骚乱分级...." << endl;
		singleRankEstimate(capture);
	}
	else
	{
		std::cout << "label值非法" << std::endl;
	}

	std::clock_t end = std::clock();
	std::cout << "耗时：" << end - start << std::endl
		<< "帧数：" << capture.get(CV_CAP_PROP_FRAME_COUNT) << std::endl
		<< "每帧处理时间：" << (end - start) / capture.get(CV_CAP_PROP_FRAME_COUNT) << std::endl;
	
	capture.release();
}

/*
 *针对长视频进行识别，也就是说程序需要每个若干帧给出一个表征前几帧行为的识别结果，
 *长视频是针对于一个视频中只存在一种行为的人为剪辑的视频来说的
*/

int gNum = 0;//已经处理的视频段数目
bool gEndFlag = false;//是否显示完毕
std::map<int, float> gMeanTime;//每段视频的平均处理时间
HANDLE hMutexgEndFlag;//控制对gEndFlag的互斥访问
HANDLE hMutexgMeanTime;//控制对给MeanTime的互斥访问

void recogLongVideo(const std::pair<std::string, cv::Size2i>& clfSz)
{
	cv::VideoCapture capture;

	//初始化视频获取源
	CCapture myCapture(clfSz.second.width, clfSz.second.height);
	if (!myCapture.initCapture(capture))
	{
		capture.release();
		return;
	}
	float label = -1;

	//std::clock_t start = std::clock();
	//加载分类器
	CvSVM svm;
	svm.load(clfSz.first.c_str());
	
	//判断是否需要对传入的视频进行格式转换 
	bool cvtFlag = false;

	const int width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
	const int height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));

	//检测视频尺寸是否跟训练SVM时使用的视频尺寸相同
	if ((width != clfSz.second.width) || (height != clfSz.second.height))
	{
		cvtFlag = true;
	}
	
	//调整视频帧率
	const double fps = capture.get(CV_CAP_PROP_FPS);
	const int delay = static_cast<int>(1000 / fps) - 10;
	
	//保存为视频
	cv::VideoWriter writer("LongVideo.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, cv::Size(320, 240));
	
	std::vector<cv::Mat> video(FRAME_NUM, cv::Mat::zeros(clfSz.second, CV_8UC1));
	std::vector<cv::Mat> colorVideo(FRAME_NUM, cv::Mat::zeros(clfSz.second, CV_32FC3));

	//创建用于显示的线程
	HANDLE hThread1;
	hThread1 = CreateThread(NULL, 0, Fun1Proc, NULL, 0, NULL);
	CloseHandle(hThread1);

	//创建互斥对象
	hMutexgEndFlag = CreateMutex(NULL, true, (LPCWSTR)("gEndFlag"));
	ReleaseMutex(hMutexgEndFlag);
	hMutexgMeanTime = CreateMutex(NULL, true, (LPCWSTR)("gMeanTime"));
	ReleaseMutex(hMutexgMeanTime);

	int vedioNum = 0; //短视频段的序列号
	while(1)
	{
		//程序耗时统计
		std::clock_t start = std::clock();
		std::clock_t end = 0;

		//获取视频帧
		if (!queryFrames(capture, FRAME_NUM, cvtFlag, video, colorVideo))
		{
			break;
		}

		vedioNum ++;
		gNum = vedioNum;

		//获取视频描述矩阵
		cv::Mat desc;
		VIF vif;
		vif.descripVideoG(video, desc);

		//描述矩阵转换为描述向量
		cv::Mat testData;
		cvtMat2Row(desc, testData);

		//使用支持向量机进行识别
		label = svm.predict(testData);

		//创建保存的短视频段
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
			cout << "骚乱分级...." << endl;
			cout << endl;
			detectFlag = true;
			longRankEstimate(colorVideo, writer, slipWriter);
		}
		else
		{
			std::cout << "label值非法" << std::endl;
		}

		end = std::clock();
		std::cout << "耗时：" << end - start << std::endl
			      << "每帧耗时：" << (end - start) / static_cast<float>(FRAME_NUM) << std::endl;

		float meanTime = (end - start) / static_cast<float>(FRAME_NUM);

		//保存每个视频段的平均处理时间
		WaitForSingleObject(hMutexgMeanTime, INFINITE);
 		gMeanTime.insert(std::pair<int, float>(vedioNum, meanTime));
		ReleaseMutex(hMutexgMeanTime);
	}

	//当主线程处理完后，等待显示线程处理完毕
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
	//判断是否有一段已经处理好的短视频段，如果没有，则一直查询等待
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
		//如果要读入的视频段未处理完，则等待
		while (gNum < loopnum)
		{
			waitCount ++;
			if (waitCount == 5)
			{
				break;
			}

			Sleep(1000);
		}

		//如果等待waitCount=5仍未处理完，则可判断整个长视频已经处理结束
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

		//读入当前要显示的视频段的平均处理时间
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

		//每读完一段视频，就删除它
		remove(str.c_str());
		loopnum++;
	}
	return 0;
}

/*
 *针对某一个人为分割好的短视频，若其为异常的，则对其异常等级进行分析
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
	CvFont font; //图片中输出字体
	cvInitFont(&font, CV_FONT_HERSHEY_TRIPLEX, 0.5, 0.5, 0, 1, CV_AA); //初始化字体

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
				//显示人群状态
				char* CrowdState = "Abnormal";
				CvSize text_size;
				cvGetTextSize(m_text, &font, &text_size, NULL);
				cvPutText(pframe, CrowdState, cvPoint(3, text_size.height + 5), &font, CV_RGB(255, 0, 255));

				//显示密度等级
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
 *针对某一长视频，每隔一段时间给出一个是否为异常行为的识别结果后，
 *若其为异常的，则对其异常等级进行分析
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
	CvFont font; //图片中输出字体
	cvInitFont(&font, CV_FONT_HERSHEY_TRIPLEX, 0.5, 0.5, 0, 1, CV_AA); //初始化字体

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
				//显示人群状态
				char* CrowdState = "Abnormal";
				CvSize text_size;
				cvGetTextSize(m_text, &font, &text_size, NULL);
				cvPutText(pframe, CrowdState, cvPoint(3, text_size.height + 5), &font, CV_RGB(255, 0, 255));

				//显示密度等级
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