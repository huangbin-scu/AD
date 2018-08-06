/*
 *�ò��ָ�����Ƶ����ѵ��
*/

#include "vif.h"
#include "TextureDensity.h"

/*
 *@brief �����Ƶ������ѵ�����õ�SVM������
 *			�ٶ���Ƶ���д���NonViolence��violence�����ļ��У��˴���Ҫ����
 *@param[in] videoSet ��Ƶ������
			 videoNum ��Ƶ����
			 videoSz ��Ƶ�ߴ�
*/
void train(const VideoSet& videoSet)
{
	//��ȡ��Ƶ����VIF����
	cv::Mat nonViolenceData;
	cv::Mat violenceData;
	readDesc(videoSet, violenceData, nonViolenceData);

	//ȷ�����ݺϷ���������������Ƶ�ĳߴ�
	if (nonViolenceData.cols != violenceData.cols)
	{
		std::cout << "�Ƿ�ѵ������---train" << std::endl;
		throw("�Ƿ�ѵ������---train");
	}

	//����ѵ�������Լ���ǩ
	cv::Mat trainData;
	cv::Mat labels;

	//nonViolence��0��ʾ��violence��1��ʾ
	cv::Mat label1 = cv::Mat::zeros(nonViolenceData.rows, 1, CV_32F);
	cv::Mat label2 = cv::Mat::ones(violenceData.rows, 1, CV_32F);

	//���ѵ�����ݺͶ�Ӧ�ı�ǩ
	trainData.push_back(nonViolenceData);
	trainData.push_back(violenceData);
	labels.push_back(label1);
	labels.push_back(label2);

	//ѵ��SVM������
	CvSVM svm = CvSVM();  
	
	//�趨����������
	CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::RBF;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	std::cout << "violence & nonViolence train" << std::endl;
	svm.train_auto(trainData, labels, cv::Mat(), cv::Mat(), params);
	std::stringstream ss;
	ss << "data/CLF/CLF-RBF" << videoSet.videosetName << ".yml";
	svm.save((ss.str()).c_str());
}

/*
 *�쳣��Ϊ�ּ�ʱ�ܶ�ѵ��
 *ly
 *2014-3-21
*/
void trainDensity()
{
	CTextureDensity* pdensity = new CTextureDensity();
	pdensity->InitialTextureDensity(TRAIN_PROCESS);
	pdensity->TrainSample();
	delete pdensity;
}