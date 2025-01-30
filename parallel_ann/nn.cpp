#include "nn.hpp"

random_device rd; // Seed
mt19937 gen(rd()); // Mersenne Twister generator
uniform_real_distribution<float> dis(0.0, 1.0);

//constructor initializes the neural network with random values for weights and biases [0, 1]
NeuralNetwork::NeuralNetwork(vector<int> topology, float learningRate):
        _topology(topology),
        _neuronMatrix({}),
        _unactivatedMatrix({}),
        _biasMatrix({}),
        _weightMatrix({}),
        _learningRate(learningRate)
        {

            for(int i=1; i < topology.size(); ++i){
                
                Matrix<float> weightMatrix(topology[i], topology[i-1]);
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
            _unactivatedMatrix.resize(topology.size());
        }

//constructor initializes the neural network with random values for weights and biases [0, 1]
NeuralNetwork::NeuralNetwork(vector<int> topology, float learningRate, const char *bias_file, const char *weight_file):
        _topology(topology),
        _neuronMatrix({}),
        _unactivatedMatrix({}),
        _biasMatrix({}),
        _weightMatrix({}),
        _learningRate(learningRate)
        {
            FILE *fp_bias = fopen(bias_file, "r");
            FILE *fp_weight = fopen(weight_file, "r");

            for(int i=1; i < topology.size(); ++i){
                
                Matrix<float> weightMatrix(topology[i], topology[i-1]);
                weightMatrix = weightMatrix.applyFunction([fp_weight](const float &val){
                        float temp;
                        fscanf(fp_weight, "%f ", &temp);
                        return temp;
                    });
                _weightMatrix.push_back(weightMatrix);

                Matrix<float> biasMatrix(topology[i], 1);
                biasMatrix = biasMatrix.applyFunction([fp_bias](const float &val){
                        float temp;
                        fscanf(fp_bias, "%f ", &temp);
                        return temp;
                    });
                _biasMatrix.push_back(biasMatrix);
            }

            fclose(fp_bias);
            fclose(fp_weight);
            _neuronMatrix.resize(topology.size());
            _unactivatedMatrix.resize(topology.size());
        }

// function to generate output from given input vector
bool NeuralNetwork::feedForword(vector<float> input){
    
    //check that the input size matches the given topology 
    assert (input.size() == _topology[0]);

    //transform input from vector to matrix
    Matrix<float> neuronMatrix(input.size(), 1);

    for(int i=0; i<input.size(); ++i){
        neuronMatrix._vals[i] = input[i];
    }

    Matrix<float> UnactivatedMatrix = neuronMatrix;

    for(int i=0; i < _weightMatrix.size(); ++i){
        _neuronMatrix.at(i) = neuronMatrix;
        _unactivatedMatrix.at(i) = UnactivatedMatrix;
       
        //z = W*a + b
        UnactivatedMatrix = neuronMatrix.multiply(_weightMatrix[i]);
        UnactivatedMatrix = UnactivatedMatrix.add(_biasMatrix[i]);

        //a' = activation_function(z)
        if(i == (_weightMatrix.size() - 1)){
            
            neuronMatrix = UnactivatedMatrix.Softmax();
        
        }else{       
            neuronMatrix = UnactivatedMatrix.applyFunction(ReLU);

            //normalise layers
            float max = neuronMatrix.max();
            //cout << "max: " << max << endl;
            if(max > 1)
                neuronMatrix = neuronMatrix.multiplyScaler(1/max);
        }

    }
    
    _neuronMatrix.at(_weightMatrix.size()) = neuronMatrix;
    _unactivatedMatrix.at(_weightMatrix.size()) = UnactivatedMatrix;

    return true;
}


// function to train with given output vector
bool NeuralNetwork::backPropagate(vector<float> targetOutput){
    
    if(targetOutput.size() != _topology.back())
        return false;

    // determine the simple error
    // error = target - output
    Matrix<float> errors(targetOutput.size(), 1);
    errors._vals = targetOutput;

    Matrix<float> neuronMatrix = _neuronMatrix.back();
    neuronMatrix = neuronMatrix.negative();

    errors = errors.add(neuronMatrix);

    // back propagating the error from output layer to input layer
    // and adjusting weights of weight matrices and bias matrics
    for(int i = _weightMatrix.size() - 1; i >= 0; i--){
        //calculating errrors for previous layer
        //for the output layer the error is the simple error = target - output
        //for other layers prevError[i] = (error[i+1] * weight[i+1].transpose()).multiplyElements(unactivatedZ[i].apply(activation_function'))
        Matrix<float> weightMatrix = _weightMatrix[i].transpose();
        Matrix<float> prevErrors = errors.multiply(weightMatrix);
        Matrix<float> dOutputs = _unactivatedMatrix[i].applyFunction(DReLU);

        prevErrors = prevErrors.multiplyElements(dOutputs);

        Matrix<float> gradients = errors.multiplyScaler(_learningRate);
        Matrix<float> weightGradients = _neuronMatrix[i].transpose().multiply(gradients);

        _biasMatrix[i] = _biasMatrix[i].addLimited(gradients);

        _weightMatrix[i] = _weightMatrix[i].addLimited(weightGradients);
        errors = prevErrors;
    }
    return true;
}

// function to retrive final output
vector<float> NeuralNetwork::getPredictions(){
    return _neuronMatrix.back()._vals;
}

void NeuralNetwork::print(const char *bias_file, const char *weight_file)
{
    FILE *fp = fopen(bias_file, "w");

    for(int i = 0; i < _biasMatrix.size(); ++i){
        //fprintf(fp, "-> \n");
        for (int y = 0; y < this->_biasMatrix[i]._rows; y++){
            for (int x = 0; x < this->_biasMatrix[i]._cols; x++){
                fprintf(fp, "%f ", this->_biasMatrix[i].at(x, y));
                //printf("%f ", this->_biasMatrix[i].at(x, y));
            }
            fprintf(fp, "\n");
            //printf("\n");
        }
    }
    fclose(fp);

    fp = fopen(weight_file, "w");

    for(int i = 0; i < _weightMatrix.size(); ++i){
        //fprintf(fp, "-> \n");
        for (int y = 0; y < this->_weightMatrix[i]._rows; y++){
            for (int x = 0; x < this->_weightMatrix[i]._cols; x++){
                fprintf(fp, "%f ", this->_weightMatrix[i].at(x, y));
                //printf("%f ", this->_weightMatrix[i].at(x, y));
            }
            fprintf(fp, "\n");
            //printf("\n");
        }
    }
    fclose(fp);
}
