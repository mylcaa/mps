#include <iostream>
#include "neural_network.hpp"
#include <vector>
#include <cstdio>

using namespace std;

int main()
{
    //input num of threads...

    //create topology of neural network
    int numLayers;
    printf("input num of layers (including the input and output layer into the count): \n");
    scanf("%d", numLayers);

    vector<int> topology(numLayers);
    for(int i=0; i<topology.size(); ++i){
        printf("input num of neurons in layer[%d]: \n", i);
        scanf("%d", numLayers);
        topology.push_back(numLayers);
    }

    //creating neural network
    NeuralNetwork nn(topology, 0.1);
    
    //input dataset
    printf("input dataset for training and testing: \n");
    scanf("%d", numLayers);
    file *fp = fopen(, 'r');
    vector<vector<float>> targetInputs = {
        {0.0f, 0.0f},
        {1.0f, 1.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f}
    }; 
    vector<vector<float>> targetOutputs = {
        {0.0f},
        {0.0f},
        {1.0f},
        {1.0f}
    };

    int epoch = 100000;
    
    //training the neural network with randomized data
    cout << "training start\n";

    for(int i = 0; i < epoch; i++)
    {
        int index = rand() % 4;
        nn.feedForword(targetInputs[index]);
        nn.backPropagate(targetOutputs[index]);
    }

    cout << "training complete\n";


    //testing the neural network
    for(vector<float> input : targetInputs){
        nn.feedForword(input);
        vector<float> preds = nn.getPredictions();
        
    }

    return 0;
}