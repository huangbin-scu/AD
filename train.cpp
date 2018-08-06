/*
 *该部分负责视频集的训练
*/

#include "vif.h"
#include "TextureDensity.h"

/*
 *@brief 针对视频集进行训练，得到SVM分类器
 *			假定视频集中存在NonViolence和violence两个文件夹，此处需要更改
 *@param[in] videoSet 视频集名称
			 videoNum 视频个数
			 videoSz 视频尺寸
*/
void train(const VideoSet& videoSet)
{
	//提取视频集的VIF描述
	cv::Mat nonViolenceData;
	cv::Mat violenceData;
	readDesc(videoSet, violenceData, nonViolenceData);

	//确保数据合法，列数表征了视频的尺寸
	if (nonViolenceData.cols != violenceData.cols)
	{
		std::cout << "非法训练数据---train" << std::endl;
		throw("非法训练数据---train");
	}

	//整合训练数据以及标签
	cv::Mat trainData;
	cv::Mat labels;

	//nonViolence以0表示；violence以1表示
	cv::Mat label1 = cv::Mat::zeros(nonViolenceData.rows, 1, CV_32F);
	cv::Mat label2 = cv::Mat::ones(violenceData.rows, 1, CV_32F);

	//填充训练数据和对应的标签
	trainData.push_back(nonViolenceData);
	trainData.push_back(violenceData);
	labels.push_back(label1);
	labels.push_back(label2);

	//训练SVM分类器
	CvSVM svm = CvSVM();  
	
	//设定分类器参数
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
 *异常行为分级时密度训练
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