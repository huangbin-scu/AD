/*
 *类VIF的实现
*/
#include "vif.h"
#include <numeric>
#include <ctime>

VIF::VIF()
{}

/*
 *@brief 将矩阵转换为行向量
*/
void cvtMat2Row(const cv::Mat& input, cv::Mat& output)
{
	if (input.data == NULL)
	{
		throw("非法输入--cvtMat2Row");
	}
	output.create(1, input.rows * input.cols, input.type());
	for (int i = 0; i != input.rows; ++i)
	{
		int index = i * input.cols; 

		input.row(i).copyTo(output.colRange(index, index + input.cols));
	}
}

/*
*尝试从硬盘读取视频集的描述，如果读取失败，使用算法计算，同时将结果保存到硬盘上
*/
void readDesc(const VideoSet& videoSet, cv::Mat& violenceData, cv::Mat& nonViolenceData)
{
	//尝试从硬盘直接读取描述矩阵
	cv::FileStorage fs;

	std::stringstream fileName;
	fileName << "data/desc/" << videoSet.videosetName << ".yml";

	//如果没有保存该项数据，计算描述矩阵，然后保存
	if (!fs.open(fileName.str(), cv::FileStorage::READ))
	{
		if (!fs.open(fileName.str(), cv::FileStorage::WRITE))
		{
			std::cout << "无法以写的方式打开" << fileName.str() << std::endl;
			throw("打开文件失败");
		}
		queryVideoSetVIF(videoSet, nonViolenceData, violenceData);

		fs.open(fileName.str(), cv::FileStorage::WRITE);
		fs << "nonVioDesc" << nonViolenceData;
		fs << "vioDesc" << violenceData;
	}
	else
	{
		fs["nonVioDesc"] >> nonViolenceData;
		fs["vioDesc"] >> violenceData;
	}			
	fs.release();
}


/*
 *@brief 将视频帧保存至Mat数组中
 *@param[in] capture 视频
 *           frameNum 要读取的视频帧数
 *			 cvtFlag 是否需要进行尺寸转换的标志
 *@param[out] frames 获得的视频帧数组
 *@return 如果读取成功，返回true
 *        如果读取失败，返回false
*/
bool queryFrames(cv::VideoCapture& capture, const int frameNum, const bool cvtFlag, 
	             std::vector<cv::Mat>& frames, std::vector<cv::Mat>& colorFrames)
{
	if ((!capture.isOpened()) || (frameNum < 0) || (frames.size() != frameNum))
	{
		return false;
	}
	
	//目标视频尺寸
	const int width = frames.at(0).cols;
	const int height = frames.at(0).rows;

	//获取视频帧
	cv::Mat img;		
	for (int i = 0; i != frameNum; ++i)
	{
		if (!capture.read(img))
		{
			std::cout << "获取视频帧失败！！！---queryFrames" << std::endl;
			return false;
		}

		cv::Mat frame;
		cv::cvtColor(img, frame, CV_BGR2GRAY);

		cv::Mat temp;
		img.copyTo(temp);

		if (cvtFlag)
		{
			cv::Mat dst;
			cv::resize(frame, dst, cv::Size2i(width, height));
			frames.at(i) = dst;
			cv::resize(img, dst, cv::Size2i(width, height));
			colorFrames.at(i) = dst;
		}
		else
		{
			frames.at(i) = frame;
			colorFrames.at(i) = temp;
		}
	}
	return true;
}


/*
 *针对人群骚乱5个测试视频集的交叉测试
*/
float crossValidation()
{
	//初始化视频集参数：视频集名称以及相应的视频个数
	std::map<std::string, int>setParams;
	setParams.insert(std::pair<std::string, int>("1", 25));
	setParams.insert(std::pair<std::string, int>("2", 25));
	setParams.insert(std::pair<std::string, int>("3", 25));
	setParams.insert(std::pair<std::string, int>("4", 24));
	setParams.insert(std::pair<std::string, int>("5", 24));

	//识别率
	float rate = 0.0f;
	
	//根据测试视频集进行循环
	for(int i = 1; i != 6; ++i)
	{
		std::cout << "开始以视频集" << i << "作为测试视频集进行处理······" << std::endl;

		//得到测试视频集名称
		std::stringstream testSet;
		testSet << i;
		
		//得到训练视频集名称
		std::vector<std::string>trainSets;
		for (int j = 1; j != 6; ++j)
		{
			std::stringstream temp;
			if (j != i)
			{
				temp << j;
				trainSets.push_back(temp.str());
			}
		}
		
		//所有视频集最终的描述矩阵
		cv::Mat nonVioDescs;
		cv::Mat vioDescs;

		std::cout << "开始提取训练视频集的描述矩阵······" << std::endl;

		//针对trainSets开始训练
		for (int m = 0; m != 4; ++m)
		{
			cv::Mat tempNonVioDesc, tempVioDesc;
			
			VideoSet videoSet(setParams.at(trainSets[m]), cv::Size2i(320, 240), trainSets[m]);
			//尝试从硬盘直接读取描述矩阵
			readDesc(videoSet, tempNonVioDesc, tempVioDesc);
			
			//验证得到的结果数据是否合法
			if (tempNonVioDesc.cols != tempVioDesc.cols)
			{
				std::cout << "非法结果，视频尺寸不统一---crossValidation" << std::endl;
				throw("非法结果---crossValidation");
			}

			//将各个视频集得到的描述矩阵整合到一起
			nonVioDescs.push_back(tempNonVioDesc);
			vioDescs.push_back(tempVioDesc);
		}

		//验证得到的结果数据是否合法
		if (nonVioDescs.rows != vioDescs.rows)
		{
			std::cout << "非法结果，视频个数不统一---crossValidation" << std::endl;
			throw("非法结果---crossValidation");
		}
		
		//整合训练数据以及标签
		cv::Mat trainData;
		cv::Mat labels;

		//nonViolence以0表示；violence以1表示
		cv::Mat label1 = cv::Mat::zeros(nonVioDescs.rows, 1, CV_32F);
		cv::Mat label2 = cv::Mat::ones(vioDescs.rows, 1, CV_32F);

		//填充训练数据和对应的标签
		trainData.push_back(nonVioDescs);
		trainData.push_back(vioDescs);
		labels.push_back(label1);
		labels.push_back(label2);

		//训练SVM分类器
		CvSVM svm = CvSVM();  
	
		//设定分类器参数
		CvSVMParams params;
		params.svm_type    = CvSVM::C_SVC;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 50, FLT_EPSILON);

		std::cout << "开始使用SVM进行训练······" << std::endl;
		svm.train_auto(trainData, labels, cv::Mat(), cv::Mat(), params);
		std::stringstream ss;
		ss << "data/CLF/CLF-For-" << i << ".yml";
		svm.save((ss.str()).c_str());
		
		//识别
		std::cout << "正在使用得到的分类器对视频集进行识别······" << std::endl;
		float tempRate = 0;
		recogVideoSet(ss.str(), VideoSet(setParams.at(testSet.str()), 
			          cv::Size2i(320, 240), testSet.str()), tempRate);
		
		//输出识别率
		std::cout << "对视频集" << i << "进行识别，得到的识别率是：\t" << tempRate << std::endl;
		
		//累计识别率
		rate += tempRate;
	}
	
	return rate / 5.0f;
}