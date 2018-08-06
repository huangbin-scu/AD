#include "TextureDensity.h"

CTextureDensity::CTextureDensity()
{
	m_PMatrixH = NULL; //0度方向上的灰度共现矩阵
	m_PMatrixRD = NULL; //45度方向上的灰度共现矩阵	
	m_PMatrixV = NULL; //90度方向上的灰度共现矩阵	
	m_PMatrixLD = NULL; //135度方向上的灰度共现矩阵	

	m_distance = 5;
	m_FilterWindowWidth = 16;
	m_GrayLayerNum = 8; //初始化设为8个灰度层，可以修改 

	m_PMatrixH = new int*[m_GrayLayerNum];
	m_PMatrixRD = new int*[m_GrayLayerNum];
	m_PMatrixV = new int*[m_GrayLayerNum];
	m_PMatrixLD = new int*[m_GrayLayerNum];	

	for (int i = 0; i < m_GrayLayerNum; i++)
	{
		m_PMatrixH[i] = new int[m_GrayLayerNum];
		m_PMatrixRD[i] = new int[m_GrayLayerNum];
		m_PMatrixV[i] = new int[m_GrayLayerNum];
		m_PMatrixLD[i] = new int[m_GrayLayerNum];
	}
}

/* 初始化基于纹理的密度估计
 */
void CTextureDensity::InitialTextureDensity(int flag)
{
	if (flag == TRAIN_PROCESS)
	{
		m_svmResponseMat = cvCreateMat(VERY_HIGH_IMAGE_NUM + HIGH_IMAGE_NUM, 1, CV_32SC1);
		m_svmDataMat = cvCreateMat(VERY_HIGH_IMAGE_NUM + HIGH_IMAGE_NUM, CHARACTER_NUM, CV_32FC1);

		m_criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON );
		m_param.svm_type = CvSVM::C_SVC;
		m_param.kernel_type = CvSVM::RBF;

		for (int i = 0; i < (HIGH_IMAGE_NUM + VERY_HIGH_IMAGE_NUM) * CHARACTER_NUM; i++)
			m_svmData[i] = -1;
	}
	else if (flag == TEST_PROCESS)
	{
		m_testMat = cvCreateMat(1, CHARACTER_NUM, CV_32FC1);
		m_svm.load("data\\CLF\\svm_veryHigh2high.xml"); 
	}
	else
	{
		cout << "Density initial failed!---InitialTextureDensity" << endl;
		throw("Density initial failed!---InitialTextureDensity");
	}
}

/* 计算四个方向的灰度共生矩阵
 * LocalImage[in]：待处理的m_FilterWindowWidth * m_FilterWindowWidth大小的图像
 */
void CTextureDensity::ComputeMatrix(IplImage* LocalImage)
{
	IplImage* NewImage = NULL;
	NewImage = cvCreateImage(cvSize(LocalImage->width, LocalImage->height), LocalImage->depth, LocalImage->nChannels);
	int i,j;

	uchar* Localdata = (uchar*)LocalImage->imageData;
	uchar* ndata = (uchar*)NewImage->imageData;
	for (i = 0; i < m_FilterWindowWidth; i++)
	{
		for (j = 0; j < m_FilterWindowWidth; j++)
		{
			//分成GrayLayerNum个灰度级
			ndata[i * NewImage->widthStep + j] = Localdata[i * LocalImage->widthStep + j] / (256 / m_GrayLayerNum);
		}
	}

	for (i = 0; i < m_GrayLayerNum; i++)
	{
		for (j = 0; j < m_GrayLayerNum; j++)
		{
			m_PMatrixH[i][j]  = 0;
			m_PMatrixLD[i][j] = 0;
			m_PMatrixRD[i][j] = 0;
			m_PMatrixV[i][j]  = 0;
		}
	}

	//计算0度的灰度共现阵
	for (i = 0; i < m_FilterWindowWidth; i++)
	{
		for (j = 0; j < m_FilterWindowWidth - m_distance; j++)
		{
			int GrayTemp1 = ndata[i * LocalImage->widthStep + j];
			int GrayTemp2 = ndata[i * LocalImage->widthStep + j + m_distance];
			m_PMatrixH[GrayTemp1][GrayTemp2] ++;
			m_PMatrixH[GrayTemp2][GrayTemp1] ++;
		}
	}

	//计算45度的灰度共现阵
	for (i = m_distance; i < m_FilterWindowWidth; i++)
	{
		for (j = 0; j < m_FilterWindowWidth - m_distance; j++)
		{
			int GrayTemp1 = ndata[i * LocalImage->widthStep + j];
			int GrayTemp2 = ndata[(i - m_distance) * LocalImage->widthStep + j + m_distance];
			m_PMatrixRD[GrayTemp1][GrayTemp2] ++;
			m_PMatrixRD[GrayTemp2][GrayTemp1] ++;
		}
	}

	//计算90度的灰度共现阵
	for (i = 0; i < m_FilterWindowWidth - m_distance; i++)
	{
		for (j = 0; j < m_FilterWindowWidth; j++)
		{
			int GrayTemp1 = ndata[i * LocalImage->widthStep + j];
			int GrayTemp2 = ndata[(i + m_distance) * LocalImage->widthStep + j];
			m_PMatrixV[GrayTemp1][GrayTemp2] ++;
			m_PMatrixV[GrayTemp2][GrayTemp1] ++;
		}
	}

	//计算135度的灰度共现阵
	for (i = 0; i < m_FilterWindowWidth - m_distance; i++)
	{
		for (j = 0; j < m_FilterWindowWidth - m_distance; j++)
		{
			int GrayTemp1 = ndata[i * LocalImage->widthStep + j];
			int GrayTemp2 = ndata[(i + m_distance) * LocalImage->widthStep + j + m_distance];
			m_PMatrixLD[GrayTemp1][GrayTemp2] ++;
			m_PMatrixLD[GrayTemp2][GrayTemp1] ++;
		}
	}
	cvReleaseImage(&NewImage);
}

/* 分别计算灰度共生矩阵的五个特征：熵，能量，相关度，局部平稳性，惯性矩
 * MatrixDirection[in]：需要计算特征的灰度共生矩阵的方向，（0度，45度，90度，135度）
 * FeatureEnergy[out]:能量
 * FeatureEntropy[out]:熵
 * FeatureInertiaQuadrature[out]:惯性矩
 * FeatureCorrelation[out]:相关度
 * FeatureLocalCalm[out]:局部平稳性
 */
void CTextureDensity::ComputeFeature(double& FeatureEnergy, double& FeatureEntropy, double& FeatureInertiaQuadrature, 
	                                 double& FeatureCorrelation, double& FeatureLocalCalm, int MatrixDirection)
{
	int i,j;
	double **pNormalizeMatrix;
	int **pGrayMatrix;
	pGrayMatrix = new int*[m_GrayLayerNum];
	pNormalizeMatrix = new double*[m_GrayLayerNum];
	for (i = 0; i < m_GrayLayerNum; i++)
	{
		pGrayMatrix[i] = new int[m_GrayLayerNum];
		pNormalizeMatrix[i] = new double[m_GrayLayerNum];
	}

	if (MatrixDirection == 0)
	{
		for (i = 0; i < m_GrayLayerNum; i++)
		{
			for (j = 0; j < m_GrayLayerNum; j++)
				pGrayMatrix[i][j] = m_PMatrixH[i][j];
		}

	}
	else if (MatrixDirection == 45)
	{
		for (i = 0; i < m_GrayLayerNum; i++)
		{
			for (j = 0; j < m_GrayLayerNum; j++)
			{
				pGrayMatrix[i][j] = m_PMatrixRD[i][j];
			}
		}
	}
	else if (MatrixDirection == 90)
	{
		for (i = 0; i < m_GrayLayerNum; i++)
		{
			for (j = 0; j < m_GrayLayerNum; j++)
			{
				pGrayMatrix[i][j] = m_PMatrixV[i][j];
			}
		}
	}
	else
	{
		for (i = 0; i < m_GrayLayerNum; i++)
		{
			for (j = 0; j < m_GrayLayerNum; j++)
			{
				pGrayMatrix[i][j] = m_PMatrixLD[i][j];
			}
		}
	}

	int total = 0;
	for (i = 0; i < m_GrayLayerNum; i++)
	{
		for (j = 0; j < m_GrayLayerNum; j++)
		{
			total += pGrayMatrix[i][j];
		}
	}

	//对灰度共生矩阵进行归一化
	for (i = 0; i < m_GrayLayerNum; i++)
	{
		for (j = 0; j < m_GrayLayerNum; j++)
		{
			pNormalizeMatrix[i][j] = (double)pGrayMatrix[i][j] / (double)total;
		}
	}

	FeatureEnergy = 0.0;
	FeatureEntropy = 0.0;
	FeatureInertiaQuadrature = 0.0;
	FeatureLocalCalm = 0.0;

	//计算能量、熵、惯性矩、局部平稳
	for (i = 0; i < m_GrayLayerNum; i++)
	{
		for (j = 0; j < m_GrayLayerNum; j++)
		{
			//能量
			FeatureEnergy += pNormalizeMatrix[i][j] * pNormalizeMatrix[i][j];
			//熵
			if (pNormalizeMatrix[i][j] > 1e-12)
				FeatureEntropy -= pNormalizeMatrix[i][j] * log(pNormalizeMatrix[i][j]);
			//惯性矩
			FeatureInertiaQuadrature += (double)(i - j) * (double)(i - j) * pNormalizeMatrix[i][j];
			//局部平稳
			FeatureLocalCalm += pNormalizeMatrix[i][j] / (1 + (double)(i - j) * (double)(i - j));
		}
	}

	//计算ux
	double ux = 0.0;
	double localtotal = 0.0;
	for (i = 0; i < m_GrayLayerNum; i++)
	{
		localtotal = 0.0;
		for (j = 0; j < m_GrayLayerNum; j++)
		{
			localtotal += pNormalizeMatrix[i][j];
		}
		ux += (double)i * localtotal;
	}

	//计算uy
	double uy = 0.0;
	for (j = 0; j < m_GrayLayerNum; j++)
	{
		localtotal = 0.0;
		for (i = 0; i < m_GrayLayerNum; i++)
		{
			localtotal += pNormalizeMatrix[i][j];
		}
		uy += (double)j * localtotal;
	}

	//计算sigmax
	double sigmax = 0.0;
	for (i = 0; i < m_GrayLayerNum; i++)
	{
		localtotal = 0.0;
		for (j = 0; j < m_GrayLayerNum; j++)
		{
			localtotal += pNormalizeMatrix[i][j];
		}
		sigmax += (double)(i - ux) * (double)(i - ux) * localtotal;
	}

	//计算sigmay
	double sigmay = 0.0;
	for (j = 0; j < m_GrayLayerNum; j++)
	{
		localtotal = 0.0;
		for (i = 0; i < m_GrayLayerNum; i++)
		{
			localtotal += pNormalizeMatrix[i][j];
		}
		sigmay += (double)(j - uy) * (double)(j - uy) * localtotal;
	}

	//计算相关
	FeatureCorrelation = 0.0;
	for (i = 0; i < m_GrayLayerNum; i++)
	{
		for (j = 0; j < m_GrayLayerNum; j++)
		{
			FeatureCorrelation += (double)(i - ux) * (double)(j - uy) * pNormalizeMatrix[i][j];
		}
	}

	if (sigmax == 0 || sigmay == 0)
	{
		FeatureCorrelation = 0.0;
		for (int i = 0; i < m_GrayLayerNum; i++)
		{
			delete pNormalizeMatrix[i];
			delete pGrayMatrix[i];
		}
		delete pNormalizeMatrix;
		delete pGrayMatrix;
		return ;
	}

	FeatureCorrelation /= sigmax;
	FeatureCorrelation /= sigmay;

	if ((FeatureCorrelation < -1e308) || (FeatureCorrelation > 1e308))
		FeatureCorrelation = 0.0;

	for (int i = 0; i < m_GrayLayerNum; i++)
	{
		delete pNormalizeMatrix[i];
		delete pGrayMatrix[i];
	}
	delete pNormalizeMatrix;
	delete pGrayMatrix;
}

/* 提取样本的四个方向的灰度共生矩阵的特征
 * pImage[in]:待提取特征的图像
 * feature[][out]:提取的特征
 */
void CTextureDensity::CollectCharacter(IplImage* pImage, double Feature[])
{
	double FeatureLocalCalm[4] = {0.0, 0.0, 0.0, 0.0}; //0度，45度，90度，135度局部平稳性
	double FeatureCorrelation[4] = {0.0, 0.0, 0.0, 0.0}; //0度，45度，90度，135度相关性
	double FeatureInertiaQuadrature[4] = {0.0, 0.0, 0.0, 0.0}; //0度，45度，90度，135度惯性矩
	double FeatureEntropy[4] = {0.0, 0.0, 0.0, 0.0}; //0度，45度，90度，135度熵
	double FeatureEnergy[4] = {0.0, 0.0, 0.0, 0.0}; //0度，45度，90度，135度能量

	double dEnergyTemp = 0.0;
	double dEntropyTemp = 0.0;
	double dInertiaQuadratureTemp = 0.0;
	double dLocalCalmTemp = 0.0;
	double dCorrelationTemp	= 0.0;

	IplImage* arLocalImage;
	arLocalImage = cvCreateImage(cvSize(m_FilterWindowWidth, m_FilterWindowWidth), pImage->depth, pImage->nChannels);
	int rolltimeH = pImage->height / m_FilterWindowWidth;
	int rolltimeW = pImage->width / m_FilterWindowWidth;
	int i, j;
	int p, q;
	uchar* Localdata = (uchar*)arLocalImage->imageData;
	uchar* idata = (uchar*)pImage->imageData;

	//将图像分成若干个窗口，计算其纹理均值
	for (i = 0; i < rolltimeH; i++)
	{
		for (j = 0; j < rolltimeW; j++)
		{
			//首先赋值给子窗口
			for (p = 0; p < m_FilterWindowWidth; p++)
			{
				for (q = 0; q < m_FilterWindowWidth; q++)
				{
					Localdata[p * arLocalImage->widthStep + q] = idata[(i * m_FilterWindowWidth + p) * pImage->widthStep 
																 + j * m_FilterWindowWidth + q];
				}
			}
			ComputeMatrix(arLocalImage);

			ComputeFeature(dEnergyTemp, dEntropyTemp, dInertiaQuadratureTemp, dCorrelationTemp, dLocalCalmTemp, 0);			
			FeatureEnergy[0] += dEnergyTemp; //能量－0度方向
			FeatureEntropy[0] += dEntropyTemp; //熵－0度方向
			FeatureInertiaQuadrature[0] += dInertiaQuadratureTemp; //惯性矩－0度方向
			FeatureCorrelation[0] += dCorrelationTemp; //相关性－0度方向
			FeatureLocalCalm[0] += dLocalCalmTemp; //局部平稳性－0度方向

			ComputeFeature(dEnergyTemp, dEntropyTemp, dInertiaQuadratureTemp, dCorrelationTemp, dLocalCalmTemp, 45);			
			FeatureEnergy[1] += dEnergyTemp; //能量－45度方向
			FeatureEntropy[1] += dEntropyTemp; //熵－45度方向
			FeatureInertiaQuadrature[1] += dInertiaQuadratureTemp; //惯性矩－45度方向
			FeatureCorrelation[1] += dCorrelationTemp; //相关性－45度方向
			FeatureLocalCalm[1] += dLocalCalmTemp; //局部平稳性－45度方向

			ComputeFeature(dEnergyTemp, dEntropyTemp, dInertiaQuadratureTemp, dCorrelationTemp, dLocalCalmTemp, 90);			
			FeatureEnergy[2] += dEnergyTemp; //能量－90度方向
			FeatureEntropy[2] += dEntropyTemp; //熵－90度方向
			FeatureInertiaQuadrature[2] += dInertiaQuadratureTemp; //惯性矩－90度方向
			FeatureCorrelation[2] += dCorrelationTemp; //相关性－90度方向
			FeatureLocalCalm[2] += dLocalCalmTemp; //局部平稳性－90度方向

			ComputeFeature(dEnergyTemp, dEntropyTemp, dInertiaQuadratureTemp, dCorrelationTemp, dLocalCalmTemp, 135);			
			FeatureEnergy[3] += dEnergyTemp; //能量－135度方向
			FeatureEntropy[3] += dEntropyTemp; //熵－135度方向
			FeatureInertiaQuadrature[3] += dInertiaQuadratureTemp; //惯性矩－135度方向
			FeatureCorrelation[3] += dCorrelationTemp; //相关性－135度方向
			FeatureLocalCalm[3] += dLocalCalmTemp; //局部平稳性－135度方向	
		}
	}

	for (i = 0; i < 4; i++)
	{
		Feature[i] = FeatureLocalCalm[i] / (rolltimeH * rolltimeW); //局部平稳性
		Feature[i + 4] = FeatureCorrelation[i] / (rolltimeH * rolltimeW); //相关性
		Feature[i + 8] = FeatureInertiaQuadrature[i] / (rolltimeH * rolltimeW); //惯性矩
		Feature[i + 12] = FeatureEntropy[i] / (rolltimeH * rolltimeW); //熵
		Feature[i + 16] = FeatureEnergy[i] / (rolltimeH * rolltimeW); //能量
	}

	//for (i = 0; i < CHARACTER_NUM; i++)
	//{
	//	cout << Feature[i] << " ";
	//	if ((i + 1) % 4 == 0)
	//		cout << endl;
	//}
	/*cout <<endl;*/

	cvReleaseImage(&arLocalImage);
}

/* 将样本特征存储在矩阵中
 * pImage[in]:待处理的灰度图像
 * pData[out]:存放每个SVM全部样本特征的数组
 */
void CTextureDensity::StoreCharacter(IplImage* pImage, double* pData)
{
	static int label = 0;
	double SigleFeature[CHARACTER_NUM];

	CollectCharacter(pImage, SigleFeature);

	if (pData[0] == -1)
	{
		label = 0;
		for (int i = 0; i < CHARACTER_NUM; i++)
		{
			pData[label + i] = SigleFeature[i];
		}
		label = label + CHARACTER_NUM;
	}
	else
	{
		for (int i = 0; i < CHARACTER_NUM; i++)
		{
			pData[label + i] = SigleFeature[i];
		}
		label = label + CHARACTER_NUM;
	}
}

/*
 * 用支持向量机训练样本
 */
void CTextureDensity::TrainSample()
{
	IplImage* SampleImage;
	char FileName[100];
	int i,j;

	cout << "train density sample......" << endl;

	for (i = 0; i < VERY_HIGH_IMAGE_NUM; i++)
	{
		sprintf(FileName, "data\\very high samples\\%d.jpg", i);
		SampleImage = cvLoadImage(FileName, CV_LOAD_IMAGE_GRAYSCALE);
		cvEqualizeHist(SampleImage, SampleImage); //直方图均衡化
		cvErode(SampleImage, SampleImage, 0, 1);
		StoreCharacter(SampleImage, m_svmData);
		m_svmResponse[i] = 1;
	}
	j = i;
	for (i = j; i < j + HIGH_IMAGE_NUM; i++)
	{
		sprintf(FileName, "data\\high samples\\%d.jpg", i - j);
		SampleImage = cvLoadImage(FileName, CV_LOAD_IMAGE_GRAYSCALE);
		cvEqualizeHist(SampleImage, SampleImage);
		cvErode(SampleImage, SampleImage, 0, 1);
		StoreCharacter(SampleImage, m_svmData);
		m_svmResponse[i] = 0;
	}

	//将数组中的数据转化到矩阵中
	for (int i = 0; i < VERY_HIGH_IMAGE_NUM + HIGH_IMAGE_NUM; i++)
	{
		for (int j = 0; j < CHARACTER_NUM; j++)
			CV_MAT_ELEM(*m_svmDataMat, float, i, j) = m_svmData[(i * CHARACTER_NUM) + j];
	}

	for (int i = 0; i < VERY_HIGH_IMAGE_NUM + HIGH_IMAGE_NUM; i++)
		CV_MAT_ELEM(*m_svmResponseMat, int, i, 0) = m_svmResponse[i];

	/********************************************输出训练样本的特征**********************************************/
	/*ofstream outfile;
	outfile.open("data/1.txt");
	if (!outfile)
	{
		cout << "outfile cant open" << endl;
		return;
	}
	int num = 0;
	for (i = 0; i < (VERY_HIGH_IMAGE_NUM + HIGH_IMAGE_NUM) * CHARACTER_NUM; i++)
	{
		outfile << m_high2medium_data[i] << " ";
		num ++;
		if (num % 20 ==0)
			outfile << endl;
	}

	outfile.close();*/
	/*************************************************************************************************************/

	m_svm.train_auto(m_svmDataMat, m_svmResponseMat, NULL, NULL, m_param);
	m_svm.save("data\\CLF\\svm_veryHigh2high.xml");
	cout << "veryHigh2high train over" << endl;
}


/* 利用训练好的SVM对视频中的目标分类
 * image[in]：待处理的当前帧彩色图像
 * 返回值：分类标记后的当前帧彩色图像
 */
char* CTextureDensity::TestVedio(IplImage* pImage)
{

	IplImage* pGrayImage = cvCreateImage(cvSize(pImage->width, pImage->height), IPL_DEPTH_8U, 1);
	cvCvtColor(pImage, pGrayImage, CV_RGB2GRAY);
	cvEqualizeHist(pGrayImage, pGrayImage);
	cvErode(pGrayImage, pGrayImage, 0, 1);

	CollectCharacter(pGrayImage, m_testData);

	//将数组中的数据转化到矩阵中
	for (int j = 0; j < CHARACTER_NUM; j++)
		CV_MAT_ELEM(*m_testMat, float, 0, j) = m_testData[j];

	int currentRank = (int)m_svm.predict(m_testMat);
	
	//根据以前帧及当前帧给出当前帧的密度等级
	char* m_text;
	m_preFrameTexture.push_back(currentRank);
	int textureHigh = 0, textureVeryHigh = 0;
	if (m_preFrameTexture.size() < JUDGE_FRAME_TEXTURE)
	{
		for (vector<bool>::iterator iter = m_preFrameTexture.begin(); 
			 iter != m_preFrameTexture.end(); ++iter)
		{
			if (*iter == true)
				textureVeryHigh++;
			else
				textureHigh++;
		}
	}
	else
	{
		for (vector<bool>::iterator iter = m_preFrameTexture.end() - JUDGE_FRAME_TEXTURE; 
			 iter != m_preFrameTexture.end(); ++iter)
		{
			if (*iter == true)
				textureVeryHigh++;
			else
				textureHigh++;
		}
	}
	if (textureHigh > textureVeryHigh)
		m_text = "Level: High";
	else
		m_text = "Level: Very High";

	return m_text;
}

CTextureDensity::~CTextureDensity()
{
	for (int i = 0; i < m_GrayLayerNum; i++)
	{
		delete m_PMatrixH[i];
		delete m_PMatrixRD[i];
		delete m_PMatrixV[i];
		delete m_PMatrixLD[i];
	}
	delete m_PMatrixH;
	delete m_PMatrixRD;
	delete m_PMatrixV;
	delete m_PMatrixLD;
}