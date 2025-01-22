#include <iostream>
#include <vector.h>
#include <matrix.hpp>

using namespace std;

class NeuralNetwork
{
    public:
        vector<int> _topology;
        vector<Matrix<float>> _neuronMatrix;
        vector<Matrix<float>> _biasMatrix;
        vector<Matrix<float>> _weightMatrix;
        float _learningRate;

    public:
        //initializing net with random values
        NeuralNetwork(vector<int> topology);
        // function to generate output from given input vector
        bool feedForword(vector<float> input);
        // function to train with given output vector
        bool backPropagate(vector<float> targetOutput);
        // function to retrive final output
        vector<float> getPredictions();
}

//ReLU activation function - Rectified Linear Unit
inline float ReLU(float val){
    return ((val <= 0)? 0 : val);
}

//Derivative ReLU 
inline float DReLU(float val){
    return ((val <= 0)? 0 : 1);
}

//Softmax activation function helper
inline float Exp(float val){
    return exp(val);
}
