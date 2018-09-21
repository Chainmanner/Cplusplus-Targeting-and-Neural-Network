#include <ctime>
#include <iostream>

#ifndef NEURALNET_H
#define NEURALNET_H

struct trainingset_t
{
	float* fInputs;
	float* fTargetOutputs;
};

// Forward declarations for neuron_t and layer_t, so that multiple neuron activation functions can be used.
struct neuron_t;
struct layer_t;

class NeuralNet
{
	public:
		layer_t* m_hInputLayer;
		layer_t** m_hHiddenLayers;
		int m_iNumberOfHiddenLayers;
		layer_t* m_hOutputLayer;
		bool m_bBiased;

	public:
		NeuralNet( layer_t* hInputLayer, layer_t** hHiddenLayers, int iHiddenLayers, layer_t* hOutputLayer, bool bBiased = false );
		~NeuralNet();

		void FwdProp( void );
		float Train( trainingset_t** hTrainingSets, int iTrainingSets, int iTrainingIterations = 1, float fLearnRate = 0.1f, float fMomentum = 0.0f, bool bVerbose = false );

		void SetInput( float* fInputValues );
		float* GetOutput( void );

		static NeuralNet* LoadNetFromFile( const char* szFilename );
		bool SaveNetToFile( const char* szFilename );

		static int GetActivationFuncID( float (*pActivationFunc)(float, bool) );
		static void* GetActivationFunc( int id );
		static float LogisticFunction( float fInput, bool bDerivative = false );
		static float LinearFunction( float fInput, bool bDerivative = false );
		static float SomeOtherFunction( float fInput, bool bDerivative = false );
		// TODO: More activation functions.
};

struct neuron_t
{
	float fVal;
	float fError;
	float* fWeights;
	float* fPrevWeightDeltas;										// Need to store change in weight for momentum.
	float (*pActivationFunc)(float, bool);
	inline void Init()
	{
		this->fVal = 0.0f;
		this->fError = 0.0f;
		this->pActivationFunc = &NeuralNet::LogisticFunction;	// Sigmoid by default.
	}
	inline void InitNeuronConnections( int iNeuronsInTheNextLayer )
	{
		this->fWeights = new float[ iNeuronsInTheNextLayer ];
		this->fPrevWeightDeltas = new float[ iNeuronsInTheNextLayer ];
		for ( int i = 0; i < iNeuronsInTheNextLayer; i++ ) this->fPrevWeightDeltas[i] = 0.0f;

		srand((unsigned)time(0));
		for ( int i = 0; i < iNeuronsInTheNextLayer; i++ )
		{
			this->fWeights[i] = ((rand() % 100) / 100.0f - 0.5f) * 2.0f;
		}
	}
};

struct layer_t
{
	float fBias;
	neuron_t** hNeurons;
	int iNeurons;
	float fLambda;
	void Init( bool bBiased, float fBias = 1.0f, float fLambda = 0.0f )
	{
		this->fLambda = fLambda;
		this->fBias = fBias;
		this->iNeurons = 0;
		if ( bBiased )
		{
			iNeurons = 1;
		}
		this->hNeurons = new neuron_t*[ iNeurons ];
		if ( bBiased )
		{
			this->hNeurons[0] = new neuron_t;
			this->hNeurons[0]->Init();
			this->hNeurons[0]->fVal = fBias;
		}
	}
	inline void InitNeurons( bool bBiased )
	{
		for ( int i = bBiased; i < iNeurons; i++ )	// Bias neuron's already initialized if bias is present.
		{
			//std::cout << iNeurons << " " << i << " INIT'D\n";
			this->hNeurons[i]->Init();
		}
	}
	inline void InitNeurons( bool bBiased, float (*pActivationFunc)(float, bool) )
	{
		for ( int i = bBiased; i < iNeurons; i++ )	// Bias neuron's already initialized if bias is present.
		{
			this->hNeurons[i]->Init();
			this->hNeurons[i]->pActivationFunc = pActivationFunc;
		}
	}
	inline void InitNeuronConnections( int iNeuronsInTheNextLayer )
	{
		for ( int i = 0; i < iNeurons; i++ )
		{
			this->hNeurons[i]->InitNeuronConnections( iNeuronsInTheNextLayer );
		}
	}
	inline void AddNeuron( neuron_t* hNeuron )
	{
		this->iNeurons++;
		neuron_t** hNewSetOfNeurons = new neuron_t*[ iNeurons ];
		for ( int i = 0; i < iNeurons - 1; i++ )
		{
			hNewSetOfNeurons[i] = hNeurons[i];
		}
		hNewSetOfNeurons[ iNeurons - 1 ] = hNeuron;
		this->hNeurons = hNewSetOfNeurons;
	}
	void PrintValues( void )
	{
		for ( int i = 0; i < this->iNeurons; i++ )
		{
			std::cout << this->hNeurons[i]->fVal << " ";
		}
		std::cout << std::endl;
	}
};

#endif