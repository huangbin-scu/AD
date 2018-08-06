#include "vif.h"

//训练主控函数
void trainMain()
{	
	try
	{
		//训练密度统计需要的分类器
		trainDensity();

		std::map<std::string, int>videoSets;
		cv::Size2i videoSz;
		
		/*
		 *下面部分为对5个视频集进行训练，结果保存为
		 *CLF1.yml CLF2.yml CLF3.yml CLF4.yml CLF5.yml 供后面识别时使用
		 */
		videoSets.insert(std::pair<std::string, int>("1", 25));
		videoSets.insert(std::pair<std::string, int>("2", 25));
		videoSets.insert(std::pair<std::string, int>("3", 25));
		videoSets.insert(std::pair<std::string, int>("4", 24));
		videoSets.insert(std::pair<std::string, int>("5", 24));

		videoSz.width = 320;
		videoSz.height = 240;
		
		std::map<std::string, int>::iterator iter;
		for (iter = videoSets.begin(); iter != videoSets.end(); ++iter)
		{			
			train(VideoSet((*iter).second, videoSz, (*iter).first));
		}
		
		system("pause");
	}
	catch(char* str)
	{
		std::cout << str << std::endl;
		system("pause");
	}
}

//对视频集进行识别测试
void recogSetMain()
{
	float recogRate;
	try
	{
		//人群骚乱识别
		cv::Size2i videoSz(320, 240);
		recogVideoSet("data/CLF/CLF-RBF5.yml", VideoSet(25, videoSz, "5"), recogRate);

		std::cout << "recogRate:\t" << recogRate << std::endl;
		system("pause");
	}
	catch(char* str)
	{
		std::cout << str << std::endl;
		system("pause");
	}
}

//识别人为分割好的短视频
void recogSingleVideoMain()
{
	try
	{
		//人群骚乱识别
		std::pair<std::string, cv::Size2i> clfSz("data/CLF/CLF-RBF6.yml", cv::Size2i(320, 240));
		recogSingleVideo(clfSz);
	}
	catch(char* str)
	{
		std::cout << str << std::endl;
		system("pause");
	}
}

/*
*对长视频进行识别，每隔若干帧给出一个识别结果
*/
void recogLongVideoMain()
{
	try
	{
		//人群骚乱识别
		std::pair<std::string, cv::Size2i> clfSz("data/CLF/CLF-RBF6.yml", cv::Size2i(320, 240));
		recogLongVideo(clfSz);
	}
	catch(char* str)
	{
		std::cout << str << std::endl;
		system("pause");
	}
}

/*
*交叉测试部分
*针对人群骚乱5个视频集的交叉实验，首先选出一个视频集作为测试视频集，其余4个作为训练视频集，得到一个识别率，
* 然后改变测试视频集以及训练视频集的组成，得到另一个识别率，如此重复5次，最后得到一个平均识别率。
*/
void crossValidationMain()
{
	float rate = crossValidation();
	std::cout << "平均识别率为" << rate << std::endl;
	system("pause");
}

/*
 *该主函数分为5大部分，根据需要使用
*/
void main()
{
	std::cout << "人群骚乱行为检测与识别" << std::endl
		      << "\t功能选择：1——>训练视频集；2——>测试视频集；" << std::endl
		      << "\t3——>视频短视频；4——>识别连续长视频；5——>人群骚乱交叉测试"
		      << std::endl;
	int funcNum = -1;
	std::cin >> funcNum;
	switch (funcNum)
	{
		case 1:
		{
			trainMain();
			break;
		}
		case 2:
		{
			recogSetMain();
			break;
		}
		case 3:
		{
			recogSingleVideoMain();
			break;
		}
		case 4:
		{
			recogLongVideoMain();
			break;
		}
		case 5:
		{
			crossValidationMain();
			break;
		}
		default:
		{
			std::cout << "\t-----非法输入----- " << std::endl;
			system("pause");
			break;
		}
	}
}