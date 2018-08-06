#include "TextureDensity.h"

CTextureDensity::CTextureDensity()
{
	m_PMatrixH = NULL; //0�ȷ����ϵĻҶȹ��־���
	m_PMatrixRD = NULL; //45�ȷ����ϵĻҶȹ��־���	
	m_PMatrixV = NULL; //90�ȷ����ϵĻҶȹ��־���	
	m_PMatrixLD = NULL; //135�ȷ����ϵĻҶȹ��־���	

	m_distance = 5;
	m_FilterWindowWidth = 16;
	m_GrayLayerNum = 8; //��ʼ����Ϊ8���ҶȲ㣬�����޸� 

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

/* ��ʼ������������ܶȹ���
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

/* �����ĸ�����ĻҶȹ�������
 * LocalImage[in]���������m_FilterWindowWidth * m_FilterWindowWidth��С��ͼ��
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
			//�ֳ�GrayLayerNum���Ҷȼ�
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

	//����0�ȵĻҶȹ�����
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

	//����45�ȵĻҶȹ�����
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

	//����90�ȵĻҶȹ�����
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

	//����135�ȵĻҶȹ�����
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

/* �ֱ����Ҷȹ������������������أ���������ضȣ��ֲ�ƽ���ԣ����Ծ�
 * MatrixDirection[in]����Ҫ���������ĻҶȹ�������ķ��򣬣�0�ȣ�45�ȣ�90�ȣ�135�ȣ�
 * FeatureEnergy[out]:����
 * FeatureEntropy[out]:��
 * FeatureInertiaQuadrature[out]:���Ծ�
 * FeatureCorrelation[out]:��ض�
 * FeatureLocalCalm[out]:�ֲ�ƽ����
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

	//�ԻҶȹ���������й�һ��
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

	//�����������ء����Ծء��ֲ�ƽ��
	for (i = 0; i < m_GrayLayerNum; i++)
	{
		for (j = 0; j < m_GrayLayerNum; j++)
		{
			//����
			FeatureEnergy += pNormalizeMatrix[i][j] * pNormalizeMatrix[i][j];
			//��
			if (pNormalizeMatrix[i][j] > 1e-12)
				FeatureEntropy -= pNormalizeMatrix[i][j] * log(pNormalizeMatrix[i][j]);
			//���Ծ�
			FeatureInertiaQuadrature += (double)(i - j) * (double)(i - j) * pNormalizeMatrix[i][j];
			//�ֲ�ƽ��
			FeatureLocalCalm += pNormalizeMatrix[i][j] / (1 + (double)(i - j) * (double)(i - j));
		}
	}

	//����ux
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

	//����uy
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

	//����sigmax
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

	//����sigmay
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

	//�������
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

/* ��ȡ�������ĸ�����ĻҶȹ������������
 * pImage[in]:����ȡ������ͼ��
 * feature[][out]:��ȡ������
 */
void CTextureDensity::CollectCharacter(IplImage* pImage, double Feature[])
{
	double FeatureLocalCalm[4] = {0.0, 0.0, 0.0, 0.0}; //0�ȣ�45�ȣ�90�ȣ�135�Ⱦֲ�ƽ����
	double FeatureCorrelation[4] = {0.0, 0.0, 0.0, 0.0}; //0�ȣ�45�ȣ�90�ȣ�135�������
	double FeatureInertiaQuadrature[4] = {0.0, 0.0, 0.0, 0.0}; //0�ȣ�45�ȣ�90�ȣ�135�ȹ��Ծ�
	double FeatureEntropy[4] = {0.0, 0.0, 0.0, 0.0}; //0�ȣ�45�ȣ�90�ȣ�135����
	double FeatureEnergy[4] = {0.0, 0.0, 0.0, 0.0}; //0�ȣ�45�ȣ�90�ȣ�135������

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

	//��ͼ��ֳ����ɸ����ڣ������������ֵ
	for (i = 0; i < rolltimeH; i++)
	{
		for (j = 0; j < rolltimeW; j++)
		{
			//���ȸ�ֵ���Ӵ���
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
			FeatureEnergy[0] += dEnergyTemp; //������0�ȷ���
			FeatureEntropy[0] += dEntropyTemp; //�أ�0�ȷ���
			FeatureInertiaQuadrature[0] += dInertiaQuadratureTemp; //���Ծأ�0�ȷ���
			FeatureCorrelation[0] += dCorrelationTemp; //����ԣ�0�ȷ���
			FeatureLocalCalm[0] += dLocalCalmTemp; //�ֲ�ƽ���ԣ�0�ȷ���

			ComputeFeature(dEnergyTemp, dEntropyTemp, dInertiaQuadratureTemp, dCorrelationTemp, dLocalCalmTemp, 45);			
			FeatureEnergy[1] += dEnergyTemp; //������45�ȷ���
			FeatureEntropy[1] += dEntropyTemp; //�أ�45�ȷ���
			FeatureInertiaQuadrature[1] += dInertiaQuadratureTemp; //���Ծأ�45�ȷ���
			FeatureCorrelation[1] += dCorrelationTemp; //����ԣ�45�ȷ���
			FeatureLocalCalm[1] += dLocalCalmTemp; //�ֲ�ƽ���ԣ�45�ȷ���

			ComputeFeature(dEnergyTemp, dEntropyTemp, dInertiaQuadratureTemp, dCorrelationTemp, dLocalCalmTemp, 90);			
			FeatureEnergy[2] += dEnergyTemp; //������90�ȷ���
			FeatureEntropy[2] += dEntropyTemp; //�أ�90�ȷ���
			FeatureInertiaQuadrature[2] += dInertiaQuadratureTemp; //���Ծأ�90�ȷ���
			FeatureCorrelation[2] += dCorrelationTemp; //����ԣ�90�ȷ���
			FeatureLocalCalm[2] += dLocalCalmTemp; //�ֲ�ƽ���ԣ�90�ȷ���

			ComputeFeature(dEnergyTemp, dEntropyTemp, dInertiaQuadratureTemp, dCorrelationTemp, dLocalCalmTemp, 135);			
			FeatureEnergy[3] += dEnergyTemp; //������135�ȷ���
			FeatureEntropy[3] += dEntropyTemp; //�أ�135�ȷ���
			FeatureInertiaQuadrature[3] += dInertiaQuadratureTemp; //���Ծأ�135�ȷ���
			FeatureCorrelation[3] += dCorrelationTemp; //����ԣ�135�ȷ���
			FeatureLocalCalm[3] += dLocalCalmTemp; //�ֲ�ƽ���ԣ�135�ȷ���	
		}
	}

	for (i = 0; i < 4; i++)
	{
		Feature[i] = FeatureLocalCalm[i] / (rolltimeH * rolltimeW); //�ֲ�ƽ����
		Feature[i + 4] = FeatureCorrelation[i] / (rolltimeH * rolltimeW); //�����
		Feature[i + 8] = FeatureInertiaQuadrature[i] / (rolltimeH * rolltimeW); //���Ծ�
		Feature[i + 12] = FeatureEntropy[i] / (rolltimeH * rolltimeW); //��
		Feature[i + 16] = FeatureEnergy[i] / (rolltimeH * rolltimeW); //����
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

/* �����������洢�ھ�����
 * pImage[in]:������ĻҶ�ͼ��
 * pData[out]:���ÿ��SVMȫ����������������
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
 * ��֧��������ѵ������
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
		cvEqualizeHist(SampleImage, SampleImage); //ֱ��ͼ���⻯
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

	//�������е�����ת����������
	for (int i = 0; i < VERY_HIGH_IMAGE_NUM + HIGH_IMAGE_NUM; i++)
	{
		for (int j = 0; j < CHARACTER_NUM; j++)
			CV_MAT_ELEM(*m_svmDataMat, float, i, j) = m_svmData[(i * CHARACTER_NUM) + j];
	}

	for (int i = 0; i < VERY_HIGH_IMAGE_NUM + HIGH_IMAGE_NUM; i++)
		CV_MAT_ELEM(*m_svmResponseMat, int, i, 0) = m_svmResponse[i];

	/********************************************���ѵ������������**********************************************/
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


/* ����ѵ���õ�SVM����Ƶ�е�Ŀ�����
 * image[in]��������ĵ�ǰ֡��ɫͼ��
 * ����ֵ�������Ǻ�ĵ�ǰ֡��ɫͼ��
 */
char* CTextureDensity::TestVedio(IplImage* pImage)
{

	IplImage* pGrayImage = cvCreateImage(cvSize(pImage->width, pImage->height), IPL_DEPTH_8U, 1);
	cvCvtColor(pImage, pGrayImage, CV_RGB2GRAY);
	cvEqualizeHist(pGrayImage, pGrayImage);
	cvErode(pGrayImage, pGrayImage, 0, 1);

	CollectCharacter(pGrayImage, m_testData);

	//�������е�����ת����������
	for (int j = 0; j < CHARACTER_NUM; j++)
		CV_MAT_ELEM(*m_testMat, float, 0, j) = m_testData[j];

	int currentRank = (int)m_svm.predict(m_testMat);
	
	//������ǰ֡����ǰ֡������ǰ֡���ܶȵȼ�
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