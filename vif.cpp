/*
 *��VIF��ʵ��
*/
#include "vif.h"
#include <numeric>
#include <ctime>

VIF::VIF()
{}

/*
 *@brief ������ת��Ϊ������
*/
void cvtMat2Row(const cv::Mat& input, cv::Mat& output)
{
	if (input.data == NULL)
	{
		throw("�Ƿ�����--cvtMat2Row");
	}
	output.create(1, input.rows * input.cols, input.type());
	for (int i = 0; i != input.rows; ++i)
	{
		int index = i * input.cols; 

		input.row(i).copyTo(output.colRange(index, index + input.cols));
	}
}

/*
*���Դ�Ӳ�̶�ȡ��Ƶ���������������ȡʧ�ܣ�ʹ���㷨���㣬ͬʱ��������浽Ӳ����
*/
void readDesc(const VideoSet& videoSet, cv::Mat& violenceData, cv::Mat& nonViolenceData)
{
	//���Դ�Ӳ��ֱ�Ӷ�ȡ��������
	cv::FileStorage fs;

	std::stringstream fileName;
	fileName << "data/desc/" << videoSet.videosetName << ".yml";

	//���û�б���������ݣ�������������Ȼ�󱣴�
	if (!fs.open(fileName.str(), cv::FileStorage::READ))
	{
		if (!fs.open(fileName.str(), cv::FileStorage::WRITE))
		{
			std::cout << "�޷���д�ķ�ʽ��" << fileName.str() << std::endl;
			throw("���ļ�ʧ��");
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
 *@brief ����Ƶ֡������Mat������
 *@param[in] capture ��Ƶ
 *           frameNum Ҫ��ȡ����Ƶ֡��
 *			 cvtFlag �Ƿ���Ҫ���гߴ�ת���ı�־
 *@param[out] frames ��õ���Ƶ֡����
 *@return �����ȡ�ɹ�������true
 *        �����ȡʧ�ܣ�����false
*/
bool queryFrames(cv::VideoCapture& capture, const int frameNum, const bool cvtFlag, 
	             std::vector<cv::Mat>& frames, std::vector<cv::Mat>& colorFrames)
{
	if ((!capture.isOpened()) || (frameNum < 0) || (frames.size() != frameNum))
	{
		return false;
	}
	
	//Ŀ����Ƶ�ߴ�
	const int width = frames.at(0).cols;
	const int height = frames.at(0).rows;

	//��ȡ��Ƶ֡
	cv::Mat img;		
	for (int i = 0; i != frameNum; ++i)
	{
		if (!capture.read(img))
		{
			std::cout << "��ȡ��Ƶ֡ʧ�ܣ�����---queryFrames" << std::endl;
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
 *�����Ⱥɧ��5��������Ƶ���Ľ������
*/
float crossValidation()
{
	//��ʼ����Ƶ����������Ƶ�������Լ���Ӧ����Ƶ����
	std::map<std::string, int>setParams;
	setParams.insert(std::pair<std::string, int>("1", 25));
	setParams.insert(std::pair<std::string, int>("2", 25));
	setParams.insert(std::pair<std::string, int>("3", 25));
	setParams.insert(std::pair<std::string, int>("4", 24));
	setParams.insert(std::pair<std::string, int>("5", 24));

	//ʶ����
	float rate = 0.0f;
	
	//���ݲ�����Ƶ������ѭ��
	for(int i = 1; i != 6; ++i)
	{
		std::cout << "��ʼ����Ƶ��" << i << "��Ϊ������Ƶ�����д�������������" << std::endl;

		//�õ�������Ƶ������
		std::stringstream testSet;
		testSet << i;
		
		//�õ�ѵ����Ƶ������
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
		
		//������Ƶ�����յ���������
		cv::Mat nonVioDescs;
		cv::Mat vioDescs;

		std::cout << "��ʼ��ȡѵ����Ƶ�����������󡤡���������" << std::endl;

		//���trainSets��ʼѵ��
		for (int m = 0; m != 4; ++m)
		{
			cv::Mat tempNonVioDesc, tempVioDesc;
			
			VideoSet videoSet(setParams.at(trainSets[m]), cv::Size2i(320, 240), trainSets[m]);
			//���Դ�Ӳ��ֱ�Ӷ�ȡ��������
			readDesc(videoSet, tempNonVioDesc, tempVioDesc);
			
			//��֤�õ��Ľ�������Ƿ�Ϸ�
			if (tempNonVioDesc.cols != tempVioDesc.cols)
			{
				std::cout << "�Ƿ��������Ƶ�ߴ粻ͳһ---crossValidation" << std::endl;
				throw("�Ƿ����---crossValidation");
			}

			//��������Ƶ���õ��������������ϵ�һ��
			nonVioDescs.push_back(tempNonVioDesc);
			vioDescs.push_back(tempVioDesc);
		}

		//��֤�õ��Ľ�������Ƿ�Ϸ�
		if (nonVioDescs.rows != vioDescs.rows)
		{
			std::cout << "�Ƿ��������Ƶ������ͳһ---crossValidation" << std::endl;
			throw("�Ƿ����---crossValidation");
		}
		
		//����ѵ�������Լ���ǩ
		cv::Mat trainData;
		cv::Mat labels;

		//nonViolence��0��ʾ��violence��1��ʾ
		cv::Mat label1 = cv::Mat::zeros(nonVioDescs.rows, 1, CV_32F);
		cv::Mat label2 = cv::Mat::ones(vioDescs.rows, 1, CV_32F);

		//���ѵ�����ݺͶ�Ӧ�ı�ǩ
		trainData.push_back(nonVioDescs);
		trainData.push_back(vioDescs);
		labels.push_back(label1);
		labels.push_back(label2);

		//ѵ��SVM������
		CvSVM svm = CvSVM();  
	
		//�趨����������
		CvSVMParams params;
		params.svm_type    = CvSVM::C_SVC;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 50, FLT_EPSILON);

		std::cout << "��ʼʹ��SVM����ѵ��������������" << std::endl;
		svm.train_auto(trainData, labels, cv::Mat(), cv::Mat(), params);
		std::stringstream ss;
		ss << "data/CLF/CLF-For-" << i << ".yml";
		svm.save((ss.str()).c_str());
		
		//ʶ��
		std::cout << "����ʹ�õõ��ķ���������Ƶ������ʶ�𡤡���������" << std::endl;
		float tempRate = 0;
		recogVideoSet(ss.str(), VideoSet(setParams.at(testSet.str()), 
			          cv::Size2i(320, 240), testSet.str()), tempRate);
		
		//���ʶ����
		std::cout << "����Ƶ��" << i << "����ʶ�𣬵õ���ʶ�����ǣ�\t" << tempRate << std::endl;
		
		//�ۼ�ʶ����
		rate += tempRate;
	}
	
	return rate / 5.0f;
}