// Test code for a basic feedforward neural network.
// NOTE: Should probably try out a better test case.
// NOTE: Should also have error-checking using finite differences. It's inefficient, but should verify accuracy.
#if 1

#include "neuralnet.h"
#include <iostream>

int main( int argc, char** argv )
{
	bool bBiased = true;

	layer_t* hInput = new layer_t;
	hInput->Init(bBiased, 1.0f, 0.0005f);
	for ( int i = 0; i < 2; i++ )
		hInput->AddNeuron( new neuron_t );
	hInput->InitNeurons(bBiased);

	layer_t** hHiddenLayers = new layer_t*[2];
	hHiddenLayers[0] = new layer_t;
	hHiddenLayers[0]->Init(bBiased, 1.0f, 0.0005f);
	for ( int i = 0; i < 3; i++ )
		hHiddenLayers[0]->AddNeuron( new neuron_t );
	hHiddenLayers[0]->InitNeurons(bBiased);
	hHiddenLayers[1] = new layer_t;
	hHiddenLayers[1]->Init(bBiased, 1.0f, 0.0005f);
	for ( int i = 0; i < 3; i++ )
		hHiddenLayers[1]->AddNeuron( new neuron_t );
	hHiddenLayers[1]->InitNeurons(bBiased);

	layer_t* hOut = new layer_t;
	hOut->Init(false, 1.0f, 0.0005f);
	for ( int i = 0; i < 3; i++ )
		hOut->AddNeuron( new neuron_t );
	hOut->InitNeurons(false);

	NeuralNet* nn = new NeuralNet(hInput, hHiddenLayers, 2, hOut, bBiased);
	
	float fInputsOne[2] = { 0.0f, 0.0f };
	float fOutputsOne[3] = { 0.0f, 0.0f, 0.0f, };
	float fInputsTwo[2] = { 0.0f, 1.0f };
	float fOutputsTwo[3] = { 0.0f, 1.0f, 1.0f };
	float fInputsThree[2] = { 1.0f, 0.0f };
	float fOutputsThree[3] = { 0.0f, 1.0f, 1.0f };
	float fInputsFour[2] = { 1.0f, 1.0f };
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
	
	nn->Train( hTrainingData, 4, 1000, 0.1f, 0.98f, false );

	nn->SetInput( fInputsOne );
	nn->FwdProp();
	std::cout << "TEST 1:\n";
	for ( int i = 0; i < 3; i++ )
		std::cout << nn->GetOutput()[i] << std::endl;
	std::cout << "\n";

	nn->SetInput( fInputsTwo );
	nn->FwdProp();
	std::cout << "TEST 2:\n";
	for ( int i = 0; i < 3; i++ )
		std::cout << nn->GetOutput()[i] << std::endl;
	std::cout << "\n";

	nn->SetInput( fInputsThree );
	nn->FwdProp();
	std::cout << "TEST 3:\n";
	for ( int i = 0; i < 3; i++ )
		std::cout << nn->GetOutput()[i] << std::endl;
	std::cout << "\n";

	nn->SetInput( fInputsFour );
	nn->FwdProp();
	std::cout << "TEST 4:\n";
	for ( int i = 0; i < 3; i++ )
		std::cout << nn->GetOutput()[i] << std::endl;
	std::cout << "\n\n";

	std::cout << nn->m_hInputLayer->hNeurons[1]->fWeights[0] << std::endl;
	std::cout << nn->m_hInputLayer->hNeurons[1]->fWeights[1] << std::endl;
	std::cout << nn->m_hInputLayer->hNeurons[1]->fWeights[2] << std::endl << std::endl;

	std::cout << nn->m_hHiddenLayers[0]->hNeurons[1]->fWeights[0] << std::endl;
	std::cout << nn->m_hHiddenLayers[0]->hNeurons[1]->fWeights[1] << std::endl;
	std::cout << nn->m_hHiddenLayers[0]->hNeurons[1]->fWeights[2] << std::endl;
	std::cout << nn->m_hHiddenLayers[0]->hNeurons[1]->fWeights[3] << std::endl;

	layer_t* hIn2 = new layer_t;
	hIn2->Init( false );
	for ( int i = 0; i < 2; i++ )
		hIn2->AddNeuron( new neuron_t );
	hIn2->InitNeurons( false );

	layer_t* hOut2 = new layer_t;
	hOut2->Init( false );
	for ( int i = 0; i < 2; i++ )
		hOut2->AddNeuron( new neuron_t );
	hOut2->InitNeurons( false, &NeuralNet::LinearFunction );

	trainingset_t** hTrain2 = new trainingset_t*[2];
	hTrain2[0] = new trainingset_t;
	hTrain2[0]->fInputs = fInputsTwo;
	hTrain2[0]->fTargetOutputs = fInputsThree;
	hTrain2[1] = new trainingset_t;
	hTrain2[1]->fInputs = fInputsThree;
	hTrain2[1]->fTargetOutputs = fInputsTwo;

	NeuralNet* nn2 = new NeuralNet( hIn2, nullptr, 0, hOut2 );

	nn2->Train( hTrain2, 2, 1000, 0.1f, 0.9f );

	std::cout << " - - - - - - - - - - - - - - - - - - - - - -\n\n";
	nn2->SetInput( fInputsOne );
	nn2->FwdProp();
	std::cout << "TEST 1:\n";
	for ( int i = 0; i < 2; i++ )
		std::cout << fInputsOne[i] << " -> " << nn2->GetOutput()[i] << std::endl;
	std::cout << "\n";

	nn2->SetInput( fInputsTwo );
	nn2->FwdProp();
	std::cout << "TEST 2:\n";
	for ( int i = 0; i < 2; i++ )
		std::cout << fInputsTwo[i] << " -> " <<  nn2->GetOutput()[i] << std::endl;
	std::cout << "\n";

	nn2->SetInput( fInputsThree );
	nn2->FwdProp();
	std::cout << "TEST 3:\n";
	for ( int i = 0; i < 2; i++ )
		std::cout << fInputsThree[i] << " -> " <<  nn2->GetOutput()[i] << std::endl;
	std::cout << "\n";

	nn2->SetInput( fInputsFour );
	nn2->FwdProp();
	std::cout << "TEST 4:\n";
	for ( int i = 0; i < 2; i++ )
		std::cout << fInputsFour[i] << " -> " <<  nn2->GetOutput()[i] << std::endl;
	std::cout << "\n";

	delete nn;
	std::system("PAUSE");
	return 0;
}
#endif