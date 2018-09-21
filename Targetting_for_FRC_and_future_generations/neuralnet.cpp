#include "neuralnet.h"
#include "assert.h"
#include <math.h>
#include <iostream>
#include <fstream>

const float e = 2.718281828459f;

NeuralNet::NeuralNet( layer_t* hInputLayer, layer_t** hHiddenLayers, int iHiddenLayers, layer_t* hOutputLayer, bool bBiased )	// Working... I think.
{
	assert_nonlethal( hInputLayer != nullptr );
	assert_nonlethal( hOutputLayer != nullptr );
	assert_nonlethal( !((hHiddenLayers == nullptr && iHiddenLayers != 0) || (hHiddenLayers != nullptr && iHiddenLayers == 0)) );
	assert_nonlethal( iHiddenLayers >= 0 );
	this->m_hInputLayer = hInputLayer;
	this->m_hHiddenLayers = hHiddenLayers;
	this->m_hOutputLayer = hOutputLayer;
	this->m_bBiased = bBiased;
	this->m_iNumberOfHiddenLayers = iHiddenLayers;

	bool bNoHiddenLayers = (hHiddenLayers == nullptr);

	m_hInputLayer->InitNeuronConnections( bNoHiddenLayers ? m_hOutputLayer->iNeurons : m_hHiddenLayers[0]->iNeurons );

	if ( !bNoHiddenLayers )
	{
		for ( int i = 0; i < iHiddenLayers; i++ )
		{
			m_hHiddenLayers[i]->InitNeuronConnections( i == iHiddenLayers - 1 ? m_hOutputLayer->iNeurons : m_hHiddenLayers[i+1]->iNeurons );
		}
	}

	if ( bBiased )	// In case we haven't already set the bias neurons to their layer's bias values, do it again.
	{
		m_hInputLayer->hNeurons[0]->fVal = m_hInputLayer->fBias;
		for ( int i = 0; i < iHiddenLayers; i++ )
			m_hHiddenLayers[i]->hNeurons[0]->fVal = m_hHiddenLayers[i]->fBias;
	}
}

NeuralNet::~NeuralNet()
{
	delete this->m_hInputLayer;
	for ( int i = 0; i < this->m_iNumberOfHiddenLayers; i++ )
		delete this->m_hHiddenLayers[i];
	delete[] this->m_hHiddenLayers;
	delete this->m_hOutputLayer;
}

void NeuralNet::FwdProp( void )	// Working... more or less.
{
	assert_nonlethal( m_hInputLayer != nullptr );
	assert_nonlethal( m_hOutputLayer != nullptr );
	assert_nonlethal( !((m_hHiddenLayers == nullptr && m_iNumberOfHiddenLayers != 0) || (m_hHiddenLayers != nullptr && m_iNumberOfHiddenLayers == 0)) );
	bool bNoHiddenLayers = ( m_hHiddenLayers == nullptr );

	if ( bNoHiddenLayers )
	{
		float fAccumulator;
		neuron_t* hCurPrevNeuron;
		for ( int i = 0; i < m_hOutputLayer->iNeurons; i++ )
		{
			fAccumulator = 0;
			hCurPrevNeuron = nullptr;
			for ( int j = 0; j < m_hInputLayer->iNeurons; j++ )
			{
				hCurPrevNeuron = m_hInputLayer->hNeurons[j];
				fAccumulator += hCurPrevNeuron->fVal * hCurPrevNeuron->fWeights[i];
			}
			m_hOutputLayer->hNeurons[i]->fVal = m_hOutputLayer->hNeurons[i]->pActivationFunc( fAccumulator, false );
		}
	}
	else
	{
		float fAccumulator;
		neuron_t* hCurPrevNeuron;
		for ( int i = m_bBiased; i < m_hHiddenLayers[0]->iNeurons; i++ )	// Input -> Hidden 1
		{
			fAccumulator = 0;
			hCurPrevNeuron = nullptr;
			for ( int j = 0; j < m_hInputLayer->iNeurons; j++ )
			{
				hCurPrevNeuron = m_hInputLayer->hNeurons[j];
				fAccumulator += hCurPrevNeuron->fVal * hCurPrevNeuron->fWeights[i];
			}
			m_hHiddenLayers[0]->hNeurons[i]->fVal = m_hHiddenLayers[0]->hNeurons[i]->pActivationFunc( fAccumulator, false );
		}
		if ( m_iNumberOfHiddenLayers > 1 )
		{
			for ( int a = 1; a < m_iNumberOfHiddenLayers; a++ )
			{
				for ( int i = m_bBiased; i < m_hHiddenLayers[a]->iNeurons; i++ )	// Hidden (a-1) -> Hidden (a)
				{
					fAccumulator = 0;
					hCurPrevNeuron = nullptr;
					for ( int j = 0; j < m_hHiddenLayers[a-1]->iNeurons; j++ )
					{
						hCurPrevNeuron = m_hHiddenLayers[a-1]->hNeurons[j];
						fAccumulator += hCurPrevNeuron->fVal * hCurPrevNeuron->fWeights[i];
					}
					m_hHiddenLayers[a]->hNeurons[i]->fVal = m_hHiddenLayers[a]->hNeurons[i]->pActivationFunc( fAccumulator, false );
				}
			}
		}
		for ( int i = 0; i < m_hOutputLayer->iNeurons; i++ )		// Last Hidden -> Output - Note how i always starts at 0, since output layer has no bias nodes.
		{
			fAccumulator = 0;
			hCurPrevNeuron = nullptr;
			for ( int j = 0; j < m_hHiddenLayers[ m_iNumberOfHiddenLayers - 1 ]->iNeurons; j++ )
			{
				hCurPrevNeuron = m_hHiddenLayers[ m_iNumberOfHiddenLayers - 1 ]->hNeurons[j];
				fAccumulator += hCurPrevNeuron->fVal * hCurPrevNeuron->fWeights[i];
			}
			m_hOutputLayer->hNeurons[i]->fVal = m_hOutputLayer->hNeurons[i]->pActivationFunc( fAccumulator, false );
		}
	}
}

// Please note that in some places this function definition might hurt your eyes.
// The equations are as such:
//	deltaWeight(i->j) = fLearnRate * (neuron i)->fVal * (neuron j)->fError + ( fMomentum * prev_deltaWeight(i->j) )							(Change in weight)
//	(some hidden neuron)->fError = ( sum: (neuron j fError) * Weight(i->j) ) * activation_derivative( (said hidden neuron)->fVal )			(Hidden neuron error)
// NOTE: Try not to set iTrainingIterations too high, and make sure to give plenty of training data, or else overfitting may occur.
// TODO: Wouldn't it be easier to have a function that backpropagates between individual layers? Same for forward propagation.
float NeuralNet::Train( trainingset_t** hTrainingSets, int iTrainingSets, int iTrainingIterations, float fLearnRate, float fMomentum, bool bVerbose )
{
	float fErrorAccumulator = 0;
	for ( int epoch = 0; epoch < iTrainingIterations; epoch++ )
	{
		for ( int k = 0; k < iTrainingSets; k++ )	// k = current training set
		{
			for ( int i = m_bBiased; i < m_hInputLayer->iNeurons; i++ )
				m_hInputLayer->hNeurons[i]->fVal = hTrainingSets[k]->fInputs[i-m_bBiased];

			this->FwdProp();

			fErrorAccumulator = 0;
			for ( int i = 0; i < m_hOutputLayer->iNeurons; i++ )	// Calculate the error at every neuron in the output layer.
			{
				fErrorAccumulator += m_hOutputLayer->hNeurons[i]->fError = (hTrainingSets[k]->fTargetOutputs[i] - m_hOutputLayer->hNeurons[i]->fVal);
			}
			if ( bVerbose ) std::cout << "Error: " << 0.5f * fErrorAccumulator * fErrorAccumulator << std::endl;

			bool bNoHiddenLayers = ( m_hHiddenLayers == nullptr );
			if ( bNoHiddenLayers )	// Output -> Input
			{
				for ( int i = 0; i < m_hInputLayer->iNeurons; i++ )			// i = current input neuron
				{
					for ( int j = 0; j < m_hOutputLayer->iNeurons; j++ )		// j = current output neuron
					{
						m_hInputLayer->hNeurons[i]->fWeights[j] +=
							(m_hInputLayer->hNeurons[i]->fPrevWeightDeltas[j] =
									fLearnRate * m_hInputLayer->hNeurons[i]->fVal * m_hOutputLayer->hNeurons[j]->fError
										+ (fMomentum * m_hInputLayer->hNeurons[i]->fPrevWeightDeltas[j]));	// Add some momentum to avoid staying at a local optimum.

						// L2 Regularization
						if (!(m_bBiased && i == 0))
							m_hInputLayer->hNeurons[i]->fWeights[j] -=
								m_hInputLayer->fLambda * m_hInputLayer->hNeurons[i]->fWeights[j];
					}
				}
			}
			else
			{
				float fTotalError;
				for ( int i = 0; i < m_hHiddenLayers[ m_iNumberOfHiddenLayers - 1 ]->iNeurons; i++ )	// i = current neuron in the last hidden layer
				{
					fTotalError = 0;
					for ( int j = 0; j < m_hOutputLayer->iNeurons; j++ )								// j = current output neuron
					{
						fTotalError += m_hOutputLayer->hNeurons[j]->fError * m_hHiddenLayers[ m_iNumberOfHiddenLayers - 1 ]->hNeurons[i]->fWeights[j];
						m_hHiddenLayers[ m_iNumberOfHiddenLayers - 1 ]->hNeurons[i]->fWeights[j] +=
							(m_hHiddenLayers[ m_iNumberOfHiddenLayers - 1 ]->hNeurons[i]->fPrevWeightDeltas[j] =
								fLearnRate * m_hHiddenLayers[ m_iNumberOfHiddenLayers - 1 ]->hNeurons[i]->fVal * m_hOutputLayer->hNeurons[j]->fError
								+ (fMomentum * m_hHiddenLayers[ m_iNumberOfHiddenLayers - 1 ]->hNeurons[i]->fPrevWeightDeltas[j]));
						// L2 Regularization
						if (!(m_bBiased && i == 0))
							m_hHiddenLayers[ m_iNumberOfHiddenLayers - 1 ]->hNeurons[i]->fWeights[j] -=
								m_hHiddenLayers[ m_iNumberOfHiddenLayers - 1 ]->fLambda * m_hHiddenLayers[ m_iNumberOfHiddenLayers - 1 ]->hNeurons[i]->fWeights[j];
					}
					m_hHiddenLayers[ m_iNumberOfHiddenLayers - 1 ]->hNeurons[i]->fError =
						fTotalError * m_hHiddenLayers[ m_iNumberOfHiddenLayers - 1 ]
							->hNeurons[i]->pActivationFunc( m_hHiddenLayers[ m_iNumberOfHiddenLayers - 1 ]->hNeurons[i]->fVal, true );
				}
				if ( m_iNumberOfHiddenLayers > 1 )	// For cases when there's more than one hidden layer.
				{
					for ( int layer = m_iNumberOfHiddenLayers - 2; layer >= 0; layer-- )	// Hidden (layer+1) -> Hidden (layer) - counter is SIGNED here or else loop is infinite.
					{
						for ( int i = 0; i < m_hHiddenLayers[layer]->iNeurons; i++ )					// i = current neuron in the current layer
						{
							fTotalError = 0;
							for ( int j = m_bBiased; j < m_hHiddenLayers[layer+1]->iNeurons; j++ )	// j = current neuron in the next layer
							{
								fTotalError += m_hHiddenLayers[layer+1]->hNeurons[j]->fError * m_hHiddenLayers[layer]->hNeurons[i]->fWeights[j];
								m_hHiddenLayers[layer]->hNeurons[i]->fWeights[j] += (m_hHiddenLayers[layer]->hNeurons[i]->fPrevWeightDeltas[j] =
									fLearnRate * m_hHiddenLayers[layer]->hNeurons[i]->fVal * m_hHiddenLayers[layer+1]->hNeurons[j]->fError
										+ (fMomentum * m_hHiddenLayers[layer]->hNeurons[i]->fPrevWeightDeltas[j]));
								// L2 Regularization
								if ( !(m_bBiased && i == 0) )
									m_hHiddenLayers[layer]->hNeurons[i]->fWeights[j] -=
										m_hHiddenLayers[layer]->fLambda * m_hHiddenLayers[layer]->hNeurons[i]->fWeights[j];
							}
							m_hHiddenLayers[layer]->hNeurons[i]->fError = fTotalError *
									m_hHiddenLayers[layer]->hNeurons[i]->pActivationFunc( m_hHiddenLayers[layer]->hNeurons[i]->fVal, true );
						}
					}
				}
				for ( int i = 0; i < m_hInputLayer->iNeurons; i++ )				// i = current input neuron
				{
					fTotalError = 0;
					for ( int j = m_bBiased; j < m_hHiddenLayers[0]->iNeurons; j++ )	// j = current neuron in the first hidden layer
					{
						fTotalError += m_hHiddenLayers[0]->hNeurons[j]->fError * m_hInputLayer->hNeurons[i]->fWeights[j];
						m_hInputLayer->hNeurons[i]->fWeights[j] += (m_hInputLayer->hNeurons[i]->fPrevWeightDeltas[j] =
							fLearnRate * m_hInputLayer->hNeurons[i]->fVal * m_hHiddenLayers[0]->hNeurons[j]->fError
								+ (fMomentum * m_hInputLayer->hNeurons[i]->fPrevWeightDeltas[j]));
						// L2 Regularization
						if (!(m_bBiased && i == 0))
							m_hInputLayer->hNeurons[i]->fWeights[j] -=
									m_hInputLayer->fLambda * m_hInputLayer->hNeurons[i]->fWeights[j];
					}
					m_hInputLayer->hNeurons[i]->fError = fTotalError *
						m_hInputLayer->hNeurons[i]->pActivationFunc( m_hInputLayer->hNeurons[i]->fVal, true );

				}
			}
		}
	}
	return 0.5f * fErrorAccumulator * fErrorAccumulator;	// Return the mean-squared error.
}

// NOTE: fInputValues MUST have the same length as the number of non-bias input neurons. This should be obvious.
void NeuralNet::SetInput( float* fInputValues )
{
	for ( int i = m_bBiased; i < m_hInputLayer->iNeurons; i++ )
		m_hInputLayer->hNeurons[i]->fVal = fInputValues[i-m_bBiased];
}

float* NeuralNet::GetOutput( void )
{
	float* fOutputs = new float[m_hOutputLayer->iNeurons];
	for ( int i = 0; i < m_hOutputLayer->iNeurons; i++ )
		fOutputs[i] = m_hOutputLayer->hNeurons[i]->fVal;
	return fOutputs;
}

NeuralNet* NeuralNet::LoadNetFromFile( const char* szFilename )	// STATIC
{
	try
	{
		std::fstream hIn;
		hIn.open( szFilename, std::fstream::in | std::fstream::binary );

		bool bBiased;			hIn.read( (char*)&bBiased, sizeof(bool) );
		int iHiddenLayers;		hIn.read( (char*)&iHiddenLayers, sizeof(int) );
		int iInputNeurons;		hIn.read( (char*)&iInputNeurons, sizeof(int) );
		int* iHiddenNeuronsPerLayer = new int[iHiddenLayers];
		for ( int i = 0; i < iHiddenLayers; i++ )
		{
			hIn.read( (char*)&iHiddenNeuronsPerLayer[i], sizeof(int) );
		}
		int iOutputNeurons;		hIn.read( (char*)&iOutputNeurons, sizeof(int) );

		// INPUT LAYER
		layer_t* pInputLayer = new layer_t;
		pInputLayer->Init( bBiased );
		for ( int i = 0; i < iInputNeurons - bBiased; i++ )
			pInputLayer->AddNeuron( new neuron_t );
		pInputLayer->InitNeurons( bBiased );
		hIn.read( (char*)&pInputLayer->fBias, sizeof(float) );
		hIn.read( (char*)&pInputLayer->fLambda, sizeof(float) );
		int iNextLayerNeurons1;	hIn.read( (char*)&iNextLayerNeurons1, sizeof(int) );
		// We'll need to replace the initialized weights at the end, since the constructor initializes the weights.
		float** fInputWeights = new float*[iInputNeurons];	// So for now, store them in a 2D array.
		int* iInputNeuronActivations = new int[iInputNeurons];
		for ( int i = 0; i < iInputNeurons; i++ )
		{
			fInputWeights[i] = new float[iNextLayerNeurons1];
			hIn.read( (char*)&iInputNeuronActivations[i], sizeof(int) );
			for ( int j = 0; j < iNextLayerNeurons1; j++ )
			{
				hIn.read( (char*)&fInputWeights[i][j], sizeof(float) );
			}
		}

		// HIDDEN LAYERS
		layer_t** pHiddenLayers = new layer_t*[iHiddenLayers];
		float*** fHiddenWeights = new float**[iHiddenLayers];
		int** iHiddenNeuronActivations = new int*[iHiddenLayers];
		int* iNextLayerNeurons2 = new int[ iHiddenLayers ];
		for ( int layer = 0; layer < iHiddenLayers; layer++ )
		{
			pHiddenLayers[layer] = new layer_t;
			pHiddenLayers[layer]->Init( bBiased );
			for ( int i = 0; i < iHiddenNeuronsPerLayer[layer] - bBiased; i++ )
				pHiddenLayers[layer]->AddNeuron( new neuron_t );
			pHiddenLayers[layer]->InitNeurons( bBiased );
			hIn.read( (char*)&pHiddenLayers[layer]->fBias, sizeof(float) );
			hIn.read( (char*)&pHiddenLayers[layer]->fLambda, sizeof(float) );
			hIn.read( (char*)&iNextLayerNeurons2[layer], sizeof(int) );

			fHiddenWeights[layer] = new float*[ iHiddenNeuronsPerLayer[layer] ];
			iHiddenNeuronActivations[layer] = new int[ iHiddenNeuronsPerLayer[layer] ];
			for ( int i = 0; i < iHiddenNeuronsPerLayer[layer]; i++ )
			{
				fHiddenWeights[layer][i] = new float[ iNextLayerNeurons2[layer] ];
				hIn.read( (char*)&iHiddenNeuronActivations[layer][i], sizeof(int) );
				for ( int j = 0; j < iNextLayerNeurons2[layer]; j++ )
				{
					hIn.read( (char*)&fHiddenWeights[layer][i][j], sizeof(float) );
				}
			}
		}

		// OUTPUT LAYER
		layer_t* pOutputLayer = new layer_t;
		pOutputLayer->Init( false );
		for ( int i = 0; i < iOutputNeurons; i++ )
			pOutputLayer->AddNeuron( new neuron_t );
		
		int* iOutputNeuronActivations = new int[iOutputNeurons];
		for ( int i = 0; i < iOutputNeurons; i++ )
		{
			hIn.read( (char*)&iOutputNeuronActivations[i], sizeof(int) );
		}

		NeuralNet* nn = new NeuralNet( pInputLayer, pHiddenLayers, iHiddenLayers, pOutputLayer, bBiased );
		// And now we update the weights and activation functions, since the constructor initializes them.
		for ( int i = 0; i < iInputNeurons; i++ )
		{
			pInputLayer->hNeurons[i]->pActivationFunc = (float(*)(float, bool))NeuralNet::GetActivationFunc( iInputNeuronActivations[i] );
			for ( int j = 0; j < iNextLayerNeurons1; j++ )
			{
				pInputLayer->hNeurons[i]->fWeights[j] = fInputWeights[i][j];
			}
		}
		delete[] fInputWeights;
		delete[] iInputNeuronActivations;
		for ( int layer = 0; layer < iHiddenLayers; layer++ )
		{
			for ( int i = 0; i < iHiddenNeuronsPerLayer[layer]; i++ )
			{
				pHiddenLayers[layer]->hNeurons[i]->pActivationFunc = (float(*)(float, bool))NeuralNet::GetActivationFunc( iHiddenNeuronActivations[layer][i] );
				for ( int j = 0; j < iNextLayerNeurons2[layer]; j++ )
				{
					pHiddenLayers[layer]->hNeurons[i]->fWeights[j] = fHiddenWeights[layer][i][j];
				}
			}
		}
		delete[] iNextLayerNeurons2;
		delete[] iHiddenNeuronsPerLayer;
		delete[] fHiddenWeights;
		delete[] iHiddenNeuronActivations;
		for ( int i = 0; i < iOutputNeurons; i++ )
		{
			pOutputLayer->hNeurons[i]->pActivationFunc = (float(*)(float, bool))NeuralNet::GetActivationFunc( iOutputNeuronActivations[i] );
		}
		delete[] iOutputNeuronActivations;

		hIn.close();
		std::cout << "Neural network data from " << szFilename << " loaded successfully!\n";
		return nn;
	}
	catch ( std::exception &e )
	{
		std::cout << "ERROR WHILE READING " << szFilename << ": " << e.what() << std::endl;
		return nullptr;
	}
}

// NOTE: Number of neurons includes the bias neuron.
// File format:
//	m_bBiased -> m_iNumberOfHiddenLayers -> (# input neurons) -> (# hidden neurons) -> ... -> (# output neurons) -> (input layer) -> (hidden layers) -> (output layer)
//		for every layer:
//			fBias -> fLambda -> (# neurons in the next layer) -> (neurons)
//				for every neuron:
//					(activation func) -> (weights)
bool NeuralNet::SaveNetToFile( const char* szFilename )
{
	try
	{
		std::fstream hOut;
		hOut.open( szFilename, std::fstream::out | std::fstream::trunc | std::fstream::binary );
		
		hOut.write( (const char*)&m_bBiased, sizeof(bool) );
		hOut.write( (const char*)&m_iNumberOfHiddenLayers, sizeof(int) );
		hOut.write( (const char*)&m_hInputLayer->iNeurons, sizeof(int) );
		for ( int i = 0; i < m_iNumberOfHiddenLayers; i++ )
			hOut.write( (const char*)&m_hHiddenLayers[i]->iNeurons, sizeof(int) );
		hOut.write( (const char*)&m_hOutputLayer->iNeurons, sizeof(int) );

		// INPUT LAYER
		hOut.write( (const char*)&m_hInputLayer->fBias, sizeof(float) );
		hOut.write( (const char*)&m_hInputLayer->fLambda, sizeof(float) );
		int iNextLayerNeurons1 = m_iNumberOfHiddenLayers == 0 ? m_hOutputLayer->iNeurons : m_hHiddenLayers[0]->iNeurons;
		hOut.write( (const char*)&iNextLayerNeurons1, sizeof(int) );
		for ( int i = 0; i < m_hInputLayer->iNeurons; i++ )
		{
			int id = NeuralNet::GetActivationFuncID( m_hInputLayer->hNeurons[i]->pActivationFunc );
			hOut.write( (const char*)&id, sizeof(int) );
			for ( int j = 0; j < iNextLayerNeurons1; j++ )
			{
				hOut.write( (const char*)&m_hInputLayer->hNeurons[i]->fWeights[j], sizeof(float) );
			}
		}

		// HIDDEN LAYERS
		for ( int layer = 0; layer < m_iNumberOfHiddenLayers; layer++ )
		{
			hOut.write( (const char*)&m_hHiddenLayers[layer]->fBias, sizeof(float) );
			hOut.write( (const char*)&m_hHiddenLayers[layer]->fLambda, sizeof(float) );
			int iNextLayerNeurons2 = (m_iNumberOfHiddenLayers - 1) - layer == 0 ? m_hOutputLayer->iNeurons : m_hHiddenLayers[layer+1]->iNeurons;
			hOut.write( (const char*)&iNextLayerNeurons2, sizeof(int) );
			for ( int i = 0; i < m_hHiddenLayers[layer]->iNeurons; i++ )
			{
				int id = NeuralNet::GetActivationFuncID( m_hHiddenLayers[layer]->hNeurons[i]->pActivationFunc );
				hOut.write( (const char*)&id, sizeof(int) );
				for ( int j = 0; j < iNextLayerNeurons2; j++ )
				{
					hOut.write( (const char*)&m_hHiddenLayers[layer]->hNeurons[i]->fWeights[j], sizeof(float) );
				}
			}
		}

		// OUTPUT LAYER
		for ( int i = 0; i < m_hOutputLayer->iNeurons; i++ )
		{
			int id = NeuralNet::GetActivationFuncID( m_hOutputLayer->hNeurons[i]->pActivationFunc );
			hOut.write( (const char*)&id, sizeof(int) );
		}

		std::cout << "Neural network saved to " << szFilename << "!\n";
		hOut.close();
	}
	catch ( std::exception &e )
	{
		std::cout << "ERROR WHILE WRITING " << szFilename << ": " << e.what() << std::endl;
		return false;
	}

	return true;
}

int NeuralNet::GetActivationFuncID( float (*pActivationFunc)(float, bool) )
{
	if ( false )	// TODO: Add some more options.
	{

	}
	
	return 0;	// Sigmoid by default.
}

void* NeuralNet::GetActivationFunc( int id )
{
	switch ( id )
	{
		case 0:
			return &NeuralNet::LogisticFunction;
		// TODO: Add more options.
		default:
			return nullptr;
	}
}

// Derivative = neuron_output * ( 1.0 - neuron_output )
// NOTE: If bDerivative is true, fInput must be the result of a logistic function.
float NeuralNet::LogisticFunction( float fInput, bool bDerivative )	// STATIC
{
	return bDerivative ? (fInput * ( 1.0f - fInput )) : (1 / ( 1 + pow(e, -fInput) ));
}

float NeuralNet::LinearFunction( float fInput, bool bDerivative )
{
	return bDerivative ? 1.0f : fInput;
}

float NeuralNet::SomeOtherFunction( float fInput, bool bDerivative ) // STATIC
{
	return bDerivative ? 1.14393f / ( cosh( 2.0f/3.0f * fInput ) * cosh( 2.0f/3.0f * fInput ) ) : 1.7159 * tanh( 2.0f/3.0f * fInput );
}

// TODO: More activation functions.