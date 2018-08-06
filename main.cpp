#include "vif.h"

//ѵ�����غ���
void trainMain()
{	
	try
	{
		//ѵ���ܶ�ͳ����Ҫ�ķ�����
		trainDensity();

		std::map<std::string, int>videoSets;
		cv::Size2i videoSz;
		
		/*
		 *���沿��Ϊ��5����Ƶ������ѵ�����������Ϊ
		 *CLF1.yml CLF2.yml CLF3.yml CLF4.yml CLF5.yml ������ʶ��ʱʹ��
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

//����Ƶ������ʶ�����
void recogSetMain()
{
	float recogRate;
	try
	{
		//��Ⱥɧ��ʶ��
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

//ʶ����Ϊ�ָ�õĶ���Ƶ
void recogSingleVideoMain()
{
	try
	{
		//��Ⱥɧ��ʶ��
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
*�Գ���Ƶ����ʶ��ÿ������֡����һ��ʶ����
*/
void recogLongVideoMain()
{
	try
	{
		//��Ⱥɧ��ʶ��
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
*������Բ���
*�����Ⱥɧ��5����Ƶ���Ľ���ʵ�飬����ѡ��һ����Ƶ����Ϊ������Ƶ��������4����Ϊѵ����Ƶ�����õ�һ��ʶ���ʣ�
* Ȼ��ı������Ƶ���Լ�ѵ����Ƶ������ɣ��õ���һ��ʶ���ʣ�����ظ�5�Σ����õ�һ��ƽ��ʶ���ʡ�
*/
void crossValidationMain()
{
	float rate = crossValidation();
	std::cout << "ƽ��ʶ����Ϊ" << rate << std::endl;
	system("pause");
}

/*
 *����������Ϊ5�󲿷֣�������Ҫʹ��
*/
void main()
{
	std::cout << "��Ⱥɧ����Ϊ�����ʶ��" << std::endl
		      << "\t����ѡ��1����>ѵ����Ƶ����2����>������Ƶ����" << std::endl
		      << "\t3����>��Ƶ����Ƶ��4����>ʶ����������Ƶ��5����>��Ⱥɧ�ҽ������"
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
			std::cout << "\t-----�Ƿ�����----- " << std::endl;
			system("pause");
			break;
		}
	}
}