// This is just a proof-of-concept neural network.
// It's rough, and it was not intended to be used by anybody that is not me.
// The real deal will be much better. Trust me on this one.

#include <iostream>
#include "neuralnet.h"
#include "assert.h"
#include <ctime>

#if 0

#define LEARN_RATE 0.1f
#define MOMENTUM 0.9f

#define BIAS true
#define IN 3
#define HIDDEN 3
#define OUT 3

neuron_t* input[IN];
neuron_t* hidden[HIDDEN];
neuron_t* output[OUT];

void Test_Init();
float Test_FwdProp();
void Test_ShowVals();
void Test_BackProp( trainingset_t** fTrainingData, int iTrainingSets );
#line 664
int main( int argc, char** argv )
{
	assert_nonlethal( 1 == 2 );
	Test_Init();
	Test_ShowVals();

	/*input[1]->fVal = 0.25f;
	input[3]->fVal = 0.25f;

	std::cout << "\n\nFWD PROP\n\n";
	Test_FwdProp();
	Test_ShowVals();*/

	float fInputsOne[3] = { 1.0f, 0.0f, 0.0f };
	float fOutputsOne[3] = { 0.0f, 0.0f, 0.0f, };
	float fInputsTwo[3] = { 1.0f, 0.0f, 1.0f };
	float fOutputsTwo[3] = { 0.0f, 1.0f, 1.0f };
	float fInputsThree[3] = { 1.0f, 1.0f, 0.0f };
	float fOutputsThree[3] = { 0.0f, 1.0f, 1.0f };
	float fInputsFour[3] = { 1.0f, 1.0f, 1.0f };
	float fOutputsFour[3] = { 1.0f, 1.0f, 0.0f };
	trainingset_t** hTrainingData = new trainingset_t*[4];

	hTrainingData[0] = new trainingset_t;
	hTrainingData[0]->fInputs = fInputsOne;
	hTrainingData[0]->fTargetOutputs = fOutputsOne;
	hTrainingData[1] = new trainingset_t;
	hTrainingData[1]->fInputs = fInputsTwo;
	hTrainingData[1]->fTargetOutputs = fOutputsTwo;
	hTrainingData[2] = new trainingset_t;
	hTrainingData[2]->fInputs = fInputsThree;
	hTrainingData[2]->fTargetOutputs = fOutputsThree;
	hTrainingData[3] = new trainingset_t;
	hTrainingData[3]->fInputs = fInputsFour;
	hTrainingData[3]->fTargetOutputs = fOutputsFour;

	for ( int i = 0; i < 100000; i++ )
		Test_BackProp( hTrainingData, 4 );

	std::cout << "BKPROP 1:\n- - - - - - - - - - - - - - - -\n";
	//input[0]->fVal = 0.0f;
	input[0+BIAS]->fVal = 0.0f;
	input[1+BIAS]->fVal = 0.0f;
	Test_FwdProp();
	Test_ShowVals();

	std::system("PAUSE");

	std::cout << "BKPROP 2:\n- - - - - - - - - - - - - - - -\n";
	//input[0]->fVal = 0.0f;
	input[0+BIAS]->fVal = 0.0f;
	input[1+BIAS]->fVal = 1.0f;
	Test_FwdProp();
	Test_ShowVals();

	std::system("PAUSE");

	std::cout << "BKPROP 3:\n- - - - - - - - - - - - - - - -\n";
	//input[0]->fVal = 0.0f;
	input[0+BIAS]->fVal = 1.0f;
	input[1+BIAS]->fVal = 0.0f;
	Test_FwdProp();
	Test_ShowVals();

	std::system("PAUSE");

	std::cout << "BKPROP 4:\n- - - - - - - - - - - - - - - -\n";
	//input[0]->fVal = 0.0f;
	input[0+BIAS]->fVal = 1.0f;
	input[1+BIAS]->fVal = 1.0f;
	Test_FwdProp();
	Test_ShowVals();

	std::system("PAUSE");

	std::cout << "- - - - - - - - - - - - - - - - - - -\n";
	for ( int i = 0; i < IN; i++ )
	{
		for ( int j = BIAS; j < HIDDEN; j++ )
		{
			std::cout << input[i]->fWeights[j] << std::endl;
		}
		std::cout << "- - - - - - - - -\n";
	}
	std::cout << "- - - - - - - - - - - - - - - - - - -\n";
	for ( int i = 0; i < HIDDEN; i++ )
	{
		for ( int j = 0; j < OUT; j++ )
		{
			std::cout << hidden[i]->fWeights[j] << std::endl;
		}
		std::cout << "- - - - - - - - -\n";
	}
	std::cout << "- - - - - - - - - - - - - - - - - - -\n";
	std::cout << "- - - - - - - - - - - - - - - - - - -\n";

	std::system("PAUSE");
	return 0;
}

void Test_Init()
{
	srand( (unsigned)time(0) );
	for ( int j = 0; j < IN; j++ )
	{
		input[j] = new neuron_t;
		input[j]->fError = 0;
		input[j]->fVal = (j == 0 && BIAS);
		input[j]->fWeights = new float[HIDDEN];
		input[j]->fPrevWeightDeltas = new float[HIDDEN];
		for ( int i = 0; i < HIDDEN; i++ )
		{
			input[j]->fWeights[i] = ((rand() % 100) / 100.0f - 0.5f) * 2.0f;
			input[j]->fPrevWeightDeltas[i] = 0;
		}
	}
	for ( int j = 0; j < HIDDEN; j++ )
	{
		hidden[j] = new neuron_t;
		hidden[j]->fError = 0;
		hidden[j]->fVal = (j == 0 && BIAS);
		hidden[j]->fWeights = new float[OUT];
		hidden[j]->fPrevWeightDeltas = new float[OUT];
		for ( int i = 0; i < OUT; i++ )
		{
			hidden[j]->fWeights[i] = ((rand() % 100) / 100.0f - 0.5f) * 2.0f;
			hidden[j]->fPrevWeightDeltas[i] = 0;
		}
	}
	for ( int j = 0; j < OUT; j++ )
	{
		output[j] = new neuron_t;
		output[j]->fError = 0;
		output[j]->fVal = 0;
	}
}

float Test_FwdProp()
{
	float x = 0;
	for ( int j = BIAS; j < HIDDEN; j++ )
	{
		float fAccumulator = 0;
		neuron_t* hCurPrevNeuron = nullptr;
		for ( int k = 0; k < IN; k++ )
		{
			hCurPrevNeuron = input[k];
			fAccumulator += hCurPrevNeuron->fVal * hCurPrevNeuron->fWeights[j];
			x += hCurPrevNeuron->fWeights[j];
			//std::cout << "L1 W " << j << "->" << k << " " << hCurPrevNeuron->fWeights[j];
		}
		//std::cout << "Hidden " << j << ": " << fAccumulator << std::endl;
		hidden[j]->fVal = NeuralNet::LogisticFunction( fAccumulator );
	}
	for ( int e = 0; e < OUT; e++ )
	{
		float fAccumulator = 0;
		neuron_t* hCurPrevNeuron = nullptr;
		for ( int i = 0; i < HIDDEN; i++ )
		{
			hCurPrevNeuron = hidden[i];
			fAccumulator += hCurPrevNeuron->fVal * hCurPrevNeuron->fWeights[e];
			x += hCurPrevNeuron->fWeights[e];
			//std::cout << "L2 W " << i << "->" << e << " " << hCurPrevNeuron->fWeights[e];
		}
		//std::cout << "Output " << e << ": " << fAccumulator << std::endl;
		output[e]->fVal = NeuralNet::LogisticFunction( fAccumulator );
	}

	return x;
}

void Test_ShowVals()
{
	std::cout << " Input: \n";
	for ( int i = 0; i < IN; i++ )
	{
		std::cout << input[i]->fVal << std::endl;
	}
	std::cout << "---------------\n Hidden: \n";
	for ( int i = 0; i < HIDDEN; i++ )
	{
		std::cout << hidden[i]->fVal << std::endl;
	}
	std::cout << "---------------\n Output: \n";
	for ( int i = 0; i < OUT; i++ )
	{
		std::cout << output[i]->fVal << std::endl;
	}
}

void Test_BackProp( trainingset_t** hTrainingData, int iTrainingSets )
{
	// BACKPROP
	for ( int k = 0; k < iTrainingSets; k++ )
	{
		for ( int i = 0; i < IN; i++ )
			input[i]->fVal = hTrainingData[k]->fInputs[i];

		float w = Test_FwdProp();

		float fErrorAccumulator = 0;
		for ( int i = 0; i < OUT; i++ )
		{
			fErrorAccumulator += output[i]->fError = (hTrainingData[k]->fTargetOutputs[i] - output[i]->fVal);// + (lambda / 2) * w * w;
		}
		//std::cout << "k = " << k << " --- " << 0.5f * fErrorAccumulator * fErrorAccumulator << std::endl;

		for ( int i = 0; i < HIDDEN; i++ )
		{
			float fTotalError = 0;
			for ( int j = 0; j < OUT; j++ )
			{
				fTotalError += output[j]->fError * hidden[i]->fWeights[j];
				//hidden[i]->fWeights[j] *= 1 - (LEARN_RATE * lambda);
				hidden[i]->fWeights[j] += hidden[i]->fPrevWeightDeltas[j] = LEARN_RATE * hidden[i]->fVal * output[j]->fError + (MOMENTUM * hidden[i]->fPrevWeightDeltas[j])
												;//- LEARN_RATE * lambda * w;
				hidden[i]->fWeights[j] -= 0.0002f * hidden[i]->fWeights[j];
			}
			hidden[i]->fError = fTotalError * NeuralNet::LogisticFunction( hidden[i]->fVal, true );// + lambda * w;
		}

		for ( int i = 0; i < IN; i++ )
		{
			for ( int j = BIAS; j < HIDDEN; j++ )
			{
				//input[i]->fWeights[j] *= 1 - (LEARN_RATE * lambda);
				input[i]->fWeights[j] += hidden[i]->fPrevWeightDeltas[j] = LEARN_RATE * input[i]->fVal * hidden[j]->fError + (MOMENTUM * input[i]->fPrevWeightDeltas[j])
										;//- LEARN_RATE * lambda * w;
				input[i]->fWeights[j] -= 0.0001f * input[i]->fWeights[j];
			}
		}
	}
}

#endif	// 0