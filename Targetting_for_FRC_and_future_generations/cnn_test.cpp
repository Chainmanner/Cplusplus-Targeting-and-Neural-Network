// IN 32x32x1 -> Conv 28x28x6 (5) -> Pool 14x14x6 (2, MAX) -> Conv 10x10x12 (5) -> Pool 5x5x12 (2, MAX) -> FC 120 -> FC 100 -> OUT 10
// Squashing func:			1.7159 * tanh( 2/3 * input )
// Squashing derivative:	1.14393 / ( cosh( 2/3 * input ) ^ 2 )

// CNN proof-of-concept, just to see if it works.
// Do not use this if you know what's good for you.
// TODO: This doesn't work right, probably because I'm not following the proper steps.
/// Pipeline:
/// Create and test convolution (valid and full)
/// Create and test convolution backpropagation
/// Create and test downscaling
/// Create and test downscaling backpropagation
/// Test fully-connected NN
/// Test fully-connected NN backpropagation
/// Test CNN to see if it crashes
/// Test CNN training
#if 0
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <fstream>
#include "neuralnet.h"
#include "assert.h"

using namespace cv;

#define LEARN_RATE 0.1f

struct img_t
{
	float** img;
	int size;
	float** weights;
	float** errors;
	bool** selected;	// For maxpooling.
	void mkVars(const int size)
	{
		this->size = size;
		selected = new bool*[size];
		for ( int i = 0; i < size; i++ )
		{
			selected[i] = new bool[size];
			for ( int j = 0; j < size; j++ )
				selected[i][j] = false;
		}
		img = new float*[size];
		for ( int i = 0; i < size; i++ )
		{
			img[i] = new float[size];
			for ( int j = 0; j < size; j++ )
			{
				img[i][j] = 0.0f;
			}
		}
		weights = new float*[size];
		for ( int i = 0; i < size; i++ )
			weights[i] = new float[size];
		for ( int i = 0; i < size; i++ )
		{
			srand((unsigned)time(0));
			for ( int j = 0; j < size; j++ )
			{
				weights[i][j] = ((rand() % 100) / 100.0f - 0.5f) * 2.0f;
			}
		}

		errors = new float*[size];
		for ( int i = 0; i < size; i++ )
			errors[i] = new float[size];
		for ( int i = 0; i < size; i++ )
		{
			srand((unsigned)time(0));
			for ( int j = 0; j < size; j++ )
			{
				errors[i][j] = 0;
			}
		}
	}
};

img_t* img1;
img_t** conv1;
img_t** img2;
img_t** mid2;
img_t** conv2;
img_t** img3;
img_t** img4;
layer_t* fc1;
layer_t* fc2;
layer_t* out;

void FwdProp();
void BackProp( float** input, float* output );
void Test_ConvFwd( img_t* img, img_t* filter, img_t* outImg, bool bFullConv = false );
void Test_ConvBkwd( img_t* outImg, img_t* filter, img_t* img );
void Test_PoolFwd( img_t* img, img_t* outImg );
void Test_PoolBkwd( img_t* outImg, img_t* img );

int main( int argc, char** argv )
{
	Mat import = imread("Debug/_visiontest4.jpg", IMREAD_GRAYSCALE);
	/*if ( import.cols != 32 || import.rows != 32 )
	{
		std::cout << import.cols << " " << import.rows << "\n";
		return -1;
	}*/

	img1 = new img_t;
	img1->mkVars(720);
	for ( int j = 0; j < 720; j++ )
	{
		for ( int i = 0; i < 720; i++ )
		{
			img1->img[i][j] = import.at<uchar>(i,j) / 255.0f;
		}
	}
	std::cout << "your OK i guess\n";

	conv1 = new img_t*[6];
	for ( int i = 0; i < 6; i++ )
	{
		conv1[i] = new img_t;
		conv1[i]->mkVars(12);
	}

	img2 = new img_t*[6];
	for ( int i = 0; i < 6; i++ )
	{
		img2[i] = new img_t;
		//img2[i]->img = Mat::zeros( 28, 28, import.type() );
		img2[i]->mkVars(718);
	}

	// Pool -> 14x14
	// ReLU

	mid2 = new img_t*[6];
	for ( int i = 0; i < 6; i++ )
	{
		mid2[i] = new img_t;
		//mid2[i]->img = Mat::zeros( 14, 14, import.type() );
		mid2[i]->mkVars(14);
	}

	conv2 = new img_t*[2];
	for ( int i = 0; i < 2; i++ )
	{
		conv2[i] = new img_t;
		conv2[i]->mkVars(3);
	}
	img3 = new img_t*[12];
	for ( int i = 0; i < 12; i++ )
	{
		img3[i] = new img_t;
		//img3[i]->img = Mat::zeros( 10, 10, import.type() );
		img3[i]->mkVars(10);
	}

	// Pool -> 5x5
	// ReLU

	img4 = new img_t*[12];
	for ( int i = 0; i < 13; i++ )
	{
		img4[i] = new img_t;
		//img4[i]->img = Mat::zeros( 5, 5, import.type() );
		img4[i]->mkVars(5);
	}

	fc1 = new layer_t;
	fc1->Init( false );
	for ( int i = 0; i < 120; i++ )
		fc1->AddNeuron( new neuron_t );
	fc1->InitNeuronConnections( 100 );
	fc2 = new layer_t;
	fc2->Init( false );
	for ( int i = 0; i < 100; i++ )
		fc2->AddNeuron( new neuron_t );
	fc2->InitNeuronConnections( 10 );

	out = new layer_t;
	out->Init( false );
	for ( int i = 0; i < 10; i++ )
		out->AddNeuron( new neuron_t );

	//std::cout << "OK\n";
	//FwdProp();
	for ( int j = 0; j < conv1[0]->size; j++ )
		for ( int i = 0; i < conv1[0]->size; i++ )
			conv1[0]->img[i][j] = 1;// / 9.0f;// / 255.0f;// / 9.0f;
	//conv1[0]->img[1][1] = 1.0f;// / 255.0f;
	/*conv1[0]->img[1][0] = -1.0f;
	conv1[0]->img[0][1] = -1.0f;
	conv1[0]->img[2][1] = -1.0f;
	conv1[0]->img[1][2] = -1.0f;
	conv1[0]->img[1][1] = 8.0f;*/
	for ( int j = 0; j < conv1[0]->size; j++ )
		for ( int i = 0; i < conv1[0]->size; i++ )
			conv1[0]->img[i][j] /= 144;
	for ( int j = 0; j < conv1[0]->size; j++ )
	{
		for ( int i = 0; i < conv1[0]->size; i++ )
			std::cout << conv1[0]->img[i][j] << " ";
		std::cout << "\n";
	}
	Test_ConvFwd(img1, conv1[0], img2[0]);
	cv::namedWindow( "TEST", cv::WINDOW_AUTOSIZE );
	Mat img1mat = cv::Mat(720, 720, CV_8U);
	Mat img2mat = cv::Mat(718, 718, CV_8U);
	Mat filtermat = cv::Mat(3, 3, CV_8U);
	for ( int j = 0; j < 3; j++ )
	{
		for ( int i = 0; i < 3; i++ )
		{
			filtermat.at<uchar>(i,j) = (int)(conv1[0]->img[i][j] * 255.0f);
		}
	}
	for ( int j = 0; j < 720; j++ )
	{
		for ( int i = 0; i < 720; i++ )
		{
			img1mat.at<uchar>(i,j) = (int)(img1->img[i][j] * 255.0f);
		}
	}
	for ( int j = 0; j < 718; j++ )
	{
		for ( int i = 0; i < 718; i++ )
		{
			img2mat.at<uchar>(i,j) = (int)(img2[0]->img[i][j] * 255.0f);
			//std::cout << img2[0]->img[i][j] * 255.0f << std::endl;
		}
	}
	Mat img3mat = cv::Mat(56, 56, CV_8U);
	Mat img4mat = cv::Mat(28, 28, CV_8U);
	//cv::filter2D(img2mat, img4mat, 1, filtermat);
	//cv::resize(img2mat, img3mat, cv::Size(56, 56) );
	cv::imshow( "TEST", img1mat );
	cv::waitKey(0);
	cv::imshow( "TEST", img2mat );
	cv::waitKey(0);
	cv::destroyWindow( "TEST" );
	imwrite( "_convtest.jpg", img2mat );

	//std::system("PAUSE");
	return 666;
}

void Test_ConvFwd( img_t* img, img_t* filter, img_t* outImg, bool bFullConv )
{
	int outSize = img->size - filter->size + 1;

	for ( int y = 0; y < outSize; y++ )
	{
		for ( int x = 0; x < outSize; x++ )
		{
			float sum = 0;
			for ( int j = 0; j < filter->size; j++ )
			{
				for ( int i = 0; i < filter->size; i++ )
				{
					sum += img->img[i+x][j+y] * filter->img[i][j];
				}
			}
			outImg->img[x][y] = min(1.0f, abs(sum));//NeuralNet::LogisticFunction(sum); //NeuralNet::SomeOtherFunction( sum );
			//if ( sum <= 0.0f ) std::cout << sum << " at " << x << " " << y << "\n";
		}
	}
}

// deltaIMG = img->errors
// deltaFILTER = filter->errors
void Test_ConvBkwd( img_t* outImg, img_t* filter, img_t* img )	// Does this work?
{
	assert_nonlethal( img->size == outImg->size + filter->size - 1 );
	//int imgSize = outImg->size + filter->size - 1;

	for ( int y = 0; y < outImg->size; y++ )
	{
		for ( int x = 0; x < outImg->size; x++ )
		{
			for ( int j = 0; j < filter->size; j++ )
			{
				for ( int i = 0; i < filter->size; i++ )
				{
					img->errors[x+i][y+j] += filter->img[i][j] * outImg->errors[x][y];
					filter->img[i][j] += img->img[x+i][y+j] * outImg->errors[x][y];
				}
			}
		}
	}
}

void Test_PoolFwd( img_t* img, img_t* outImg )
{
	assert_nonlethal( outImg->size == img->size / 2 );

	for ( int y = 0; y < outImg->size; y += 2 )
	{
		for ( int x = 0; x < outImg->size; x += 2 )
		{
			float curMax = INT32_MIN;
			int maxX = 0;
			int maxY = 0;
			for ( int j = 0; j < 2; j++ )
			{
				for ( int i = 0; i < 2; i++ )
				{
					if ( img->img[x+i][y+j] > curMax )
					{
						curMax = img->img[x+i][y+j];
						maxX = i;
						maxY = j;
					}
				}
			}
			img->selected[x+maxX][y+maxY] = true;
			outImg->img[x / 2][y / 2] = img->img[x+maxX][y+maxY];
		}
	}
}

void Test_PoolBkwd( img_t* outImg, img_t* img )
{
	assert_nonlethal( img->size == outImg->size * 2 );

	for ( int y = 0; y < img->size; y++ )
	{
		for ( int x = 0; x < img->size; x++ )
		{
			for ( int j = 0; j < 2; j++ )
			{
				bool bShouldBreak = false;
				for ( int i = 0; i < 2; i++ )
				{
					if ( img->selected[x*2+i][y*2+j] )
					{
						bShouldBreak = true;
						img->errors[x*2+i][y*2+j] = outImg->errors[x][y];
						break;
					}
				}
				if ( bShouldBreak ) break;
			}
		}
	}
}

void FwdProp()
{
	/*for ( int j = 0; j < img1->img.rows; j++ )
	{
		for ( int i = 0; i < img1->img.cols; i++ )
		{
			img1->img.at<uchar>(i,j) /= 1; //(img1->img.at<uchar>(i,j) / 255.0f) * 1.275f - 0.1f;
		}
	}*/

	for ( int img = 0; img < 6; img++ )
	{
		for ( int i = 0; i < 28; i++ )
		{
			for ( int j = 0; j < 28; j++ )
				img2[img]->selected[i][j] = false;
		}
	}
	for ( int img = 0; img < 12; img++ )
	{
		for ( int i = 0; i < 10; i++ )
		{
			for ( int j = 0; j < 10; j++ )
				img3[img]->selected[i][j] = false;
		}
	}

	for ( int i = 0; i < 6; i++ )
	{
		Test_ConvFwd( img1, conv1[i], img2[i] );
		Test_PoolFwd( img2[i], mid2[i] );
		for ( int j = 0; j < 2; j++ )
		{
			Test_ConvFwd( mid2[i], conv2[j], img3[i*2+j] );
			Test_PoolFwd( img3[i*2+j], img4[i*2+j] );
		}
	}

	for ( int j = 0; j < 120; j++ )
	{
		float fAccumulator = 0;
		for ( int img = 0; img < 12; img++ )
		{
			for ( int y = 0; y < 5; y++ )
			{
				for ( int x = 0; x < 5; x++ )
				{
					fAccumulator += img4[img]->img[x][y] * img4[img]->weights[x][y];
				}
			}
		}
		fc1->hNeurons[j]->fVal = NeuralNet::LogisticFunction( fAccumulator );
	}

	for ( int j = 0; j < 100; j++ )
	{
		float fAccumulator = 0;
		for ( int i = 0; i < 120; i++ )
		{
			fAccumulator += fc1->hNeurons[i]->fVal * fc1->hNeurons[i]->fWeights[j];
		}
		fc2->hNeurons[j]->fVal = NeuralNet::LogisticFunction( fAccumulator );
	}

	for ( int j = 0; j < 10; j++ )
	{
		float fAccumulator = 0;
		for ( int i = 0; i < 100; i++ )
		{
			fAccumulator += fc2->hNeurons[i]->fVal * fc2->hNeurons[i]->fWeights[j];
		}
		out->hNeurons[j]->fVal = fAccumulator;//NeuralNet::LogisticFunction( fAccumulator );
	}
}

void BackProp( float** input, float* output )
{
	for ( int j = 0; j < 32; j++ )
	{
		for ( int i = 0; i < 32; i++ )
			img1->img[i][j] = input[i][j];
	}

	FwdProp();

	float fError = 0;
	for ( int i = 0; i < 10; i++ )
	{
		fError += out->hNeurons[i]->fError = out->hNeurons[i]->fVal - output[i];
	}
	std::cout << 0.5f * fError * fError << "\n";
	
	for ( int i = 0; i < 100; i++ )
	{
		float accumulator = 0;
		for ( int j = 0; j < 10; j++ )
		{
			accumulator += out->hNeurons[j]->fError * fc2->hNeurons[i]->fWeights[j];
			fc2->hNeurons[i]->fWeights[j] += out->hNeurons[j]->fError * fc2->hNeurons[i]->fVal * LEARN_RATE;
		}
		fc2->hNeurons[i]->fError = accumulator * NeuralNet::LogisticFunction( fc2->hNeurons[i]->fVal, true );
	}
	for ( int i = 0; i < 120; i++ )
	{
		float accumulator = 0;
		for ( int j = 0; j < 100; j++ )
		{
			accumulator += fc2->hNeurons[j]->fError * fc1->hNeurons[i]->fWeights[j];
			fc1->hNeurons[i]->fWeights[j] += fc2->hNeurons[j]->fError * fc1->hNeurons[i]->fVal * LEARN_RATE;
		}
		fc1->hNeurons[i]->fError = accumulator * NeuralNet::LogisticFunction( fc1->hNeurons[i]->fVal, true );
	}

	// TODO
}
#endif