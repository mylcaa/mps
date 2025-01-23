#include <iostream>
#include "nn.hpp"
#include <vector>
#include <cstdio>
#include <string>

using namespace std;

double* loadVec(char* fname, int n){
    FILE *f = fopen(fname, "r");
    double *res = new double[n];
    double *it = res;
    while(fscanf(f, "%lf", it++) != EOF);
    fclose (f);
    return res;
}

int main(int argc, char* argv[])
{
    //input num of threads...

    //create topology of neural network
    int numLayers;
    int num;
    printf("input num of layers (including the input and output layer into the count): \n");
    scanf("%d", &numLayers);

    vector<int> topology(numLayers);
    for(int i=0; i<numLayers; ++i){
        printf("input num of neurons in layer[%d]: \n", i);
        scanf("%d", &num);
        topology.push_back(num);
    }

    //creating neural network
    NeuralNetwork nn(topology, 0.1);
    
    //dataset
    //string test = argv[1];
    string test = "/home/koshek/Desktop/MS/zadaci/zad1/test/test_";
    //string train = argv[2];
    string train = "/home/koshek/Desktop/MS/zadaci/zad1/train/train_";

    int epoch = 100000;
    //0 1 2 3 4 5 6 7 8 9
    vector<float> targetOutput;
    //MNIST dataset pictures are 28x28
    vector<float> targetInput;
    
    //training the neural network with randomized data
    cout << "training start\n";

    for(int i = 0; i < epoch; i++)
    {
        int index = rand() % 60000;
        
        train.append(to_string(index));
        train.append(".txt");
        const char * ctest = train.c_str();

        FILE *fp = fopen(ctest, "r");
        assert(fp != NULL);
        
        for(int j = 0; j < 794; ++j){
            if(j<10){
                fscanf(fp, "%d ", &index);
                targetOutput.push_back(index);
            }else{
                fscanf(fp, "%d ", &index);
                targetInput.push_back(index);
            }
        }
        fclose(fp);

        nn.feedForword(targetInput);
        nn.backPropagate(targetOutput);
    }

    cout << "training complete\n";


    //testing the neural network
    /*for(vector<float> input : targetInputs){
        nn.feedForword(input);
        vector<float> preds = nn.getPredictions();
        
    }*/

    return 0;
}