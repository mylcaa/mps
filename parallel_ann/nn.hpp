#include <iostream>
#include <vector>
#include <math.h>
#include "matrix.hpp"

using namespace std;

class NeuralNetwork
{
    public:
        vector<int> _topology;
        vector<Matrix<float>> _neuronMatrix;
        vector<Matrix<float>> _unactivatedMatrix;
        vector<Matrix<float>> _biasMatrix;
        vector<Matrix<float>> _weightMatrix;
        float _learningRate;

    public:
        //initializing net with random values
        NeuralNetwork(vector<int> topology, float learningRate);
        //initializing net with trained values
        NeuralNetwork(vector<int> topology, float learningRate, const char *bias_file, const char *weight_file);
        // function to generate output from given input vector
        bool feedForword(vector<float> input);
        // function to train with given output vector
        bool backPropagate(vector<float> targetOutput);
        // function to retrive final output
        vector<float> getPredictions();
        // write down weights and biases of a trained neural network
        void print(const char *bias_file, const char *weight_file);

};

//ReLU activation function - Rectified Linear Unit
inline float ReLU(float val){
    return ((val <= 0)? 0 : val);
}

//Derivative ReLU 
inline float DReLU(float val){
    return ((val <= 0)? 0 : 1);
}
