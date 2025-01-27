#include <iostream>
#include "nn.hpp"
#include <vector>
#include <cstdio>
#include <string>

using namespace std;

NeuralNetwork train(vector<int>);
void WriteNeuralNetwork(vector<Matrix<float>>, vector<Matrix<float>>);

int main(int argc, char* argv[])
{
    //input num of threads...

    //create topology of neural network
    int numLayers;
    int num;
    printf("input num of layers (including the input and output layer into the count): \n");
    scanf("%d", &numLayers);

    vector<int> topology;
    for(int i=0; i<numLayers; ++i){
        printf("input num of neurons in layer[%d]: \n", i);
        scanf("%d", &num);
        topology.push_back(num);
    }

    NeuralNetwork nn = train(topology);

    const char *weight_file = "weights.txt";
    const char *bias_file = "biases.txt";
    nn.print(weight_file, bias_file);

    //testing the neural network
    /*for(vector<float> input : targetInputs){
        nn.feedForword(input);
        vector<float> preds = nn.getPredictions();
        
    }*/

    return 0;
}

NeuralNetwork train(vector<int> topology){

    //creating neural network
    NeuralNetwork nn(topology, 0.1);

    /*const char *weight_file = "weights.txt";
    const char *bias_file = "biases.txt";
    nn.print(weight_file, bias_file);*/
    
    //dataset
    //string test = argv[1];
    //string test = "/home/koshek/Desktop/MS/zadaci/zad1/test/test_";
    //string train = argv[2];
    
    string train;
    int epoch = 100000;
    //0 1 2 3 4 5 6 7 8 9
    vector<float> targetOutput(10);
    //MNIST dataset pictures are 28x28
    vector<float> targetInput(784);
    
    //training the neural network with randomized data
    cout << "training start\n";

    for(int i = 0; i < epoch; i++)
    {
        int index = rand() % 60000;
        
        train = "/home/koshek/Desktop/MS/zadaci/zad1/train/train_";
        train.append(to_string(index));
        train.append(".txt");
        const char * ctest = train.c_str();
    
        FILE *fp = fopen(ctest, "r");
        assert(fp != NULL);
        
        for(int j = 0; j < 794; ++j){
            if(j<10){
                fscanf(fp, "%d ", &index);
                targetOutput.at(j) = index;
            }else{
                fscanf(fp, "%d ", &index);
                targetInput.at(j-10) = index;
            }
        }
        fclose(fp);

        cout << "Feed forward" << endl;
        nn.feedForword(targetInput);

        /*const char *weight_file = "weights.txt";
        const char *bias_file = "biases.txt";
        nn.print(weight_file, bias_file);*/

        cout << "Back propagate" << endl;
        nn.backPropagate(targetOutput);

        break;
    }

    cout << "training complete\n";

    return nn;
}