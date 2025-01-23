#include "nn.hpp"
#include <random>

// Initialize a random number generator
random_device rd; // Seed
mt19937 gen(rd()); // Mersenne Twister generator
uniform_real_distribution<float> dis(0.0, 1.0);

//constructor initializes the neural network with random values for weights and biases [0, 1]
NeuralNetwork::NeuralNetwork(vector<int> topology, float learningRate):
        _topology(topology),
        _neuronMatrix({}),
        _biasMatrix({}),
        _weightMatrix({}),
        _learningRate(learningRate)
        {

            for(int i=1; i <= topology.size(); ++i){
                
                Matrix<float> weightMatrix(topology[i-1], topology[i]);
                weightMatrix = weightMatrix.applyFunction([](const float &val){
                        return dis(gen);
                    });
                _weightMatrix.push_back(weightMatrix);

                Matrix<float> biasMatrix(topology[i], 1);
                biasMatrix = biasMatrix.applyFunction([](const float &val){
                        return dis(gen);
                    });
                _biasMatrix.push_back(biasMatrix);
            }
            _neuronMatrix.resize(topology.size());
        }

// function to generate output from given input vector
bool NeuralNetwork::feedForword(vector<float> input){
    
    //check that the input size matches the given topology
    if(input.size() != _topology[0])
        return false;

    //transform input from vector to matrix
    Matrix<float> neuronMatrix(input.size(), 1);
    for(int i=0; i<input.size(); ++i)
        neuronMatrix._vals[i] = input[i];

    for(int i=0; i < _topology.size(); ++i){
        _neuronMatrix.push_back(neuronMatrix);

        //a' = activation_function(W*a + b)
        if(i == (_topology.size() - 1)){
            Matrix<float> temp = _weightMatrix[i].multiply(neuronMatrix);
            temp = temp.add(_biasMatrix[i]);
            temp = temp.applyFunction(Exp);

            float sum = temp.sumElements();
            sum = 1/sum;
            neuronMatrix = temp.multiplyScaler(sum);
        }
        else{
            neuronMatrix = _weightMatrix[i].multiply(neuronMatrix);
            neuronMatrix = neuronMatrix.add(_biasMatrix[i]);
            neuronMatrix = neuronMatrix.applyFunction(ReLU);
        }
    }
    _neuronMatrix.push_back(neuronMatrix);
    return true;
}


// function to train with given output vector
bool NeuralNetwork::backPropagate(vector<float> targetOutput){
    
    if(targetOutput.size() != _topology.back())
        return false;

    // determine the simple error
    // error = target - output
    cout << "target output size: " << targetOutput.size() << endl;
    Matrix<float> errors(targetOutput.size(), 1);
    errors._vals = targetOutput;

    Matrix<float> neuronMatrix = _neuronMatrix.back();
    neuronMatrix = neuronMatrix.negative();
    cout << "neuronMatrix size: " << neuronMatrix._cols << " " << neuronMatrix._rows << endl;
    cout << "neuronMatrix size from origigi: " << _neuronMatrix.front()._cols << " " << _neuronMatrix.front()._rows << endl;

    errors = errors.add(neuronMatrix);

    cout << "10" << endl;

    // back propagating the error from output layer to input layer
    // and adjusting weights of weight matrices and bias matrics
    for(int i = _weightMatrix.size() - 1; i >= 0; i--){
        //calculating errrors for previous layer
        Matrix<float> weightMatrix = _weightMatrix[i].transpose();
        Matrix<float> prevErrors = errors.multiply(weightMatrix);

        //calculating gradient i.e. delta weight (dw)
        //dw = lr * error * d/dx(activated value)
        Matrix<float> dOutputs = _neuronMatrix[i + 1].applyFunction(DReLU);
        Matrix<float> gradients = errors.multiplyElements(dOutputs);
        gradients = gradients.multiplyScaler(_learningRate);
        Matrix<float> weightGradients = _neuronMatrix[i].transpose().multiply(gradients);
        
        //adjusting bias and weight
        _biasMatrix[i] = _biasMatrix[i].add(gradients);
        _weightMatrix[i] = _weightMatrix[i].add(weightGradients);
        errors = prevErrors;
    }
    return true;
}

// function to retrive final output
vector<float> NeuralNetwork::getPredictions(){
    return _neuronMatrix.back()._vals;
}