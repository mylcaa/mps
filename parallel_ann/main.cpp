#include <iostream>
#include "nn.hpp"
#include <vector>
#include <cstdio>
#include <string>

using namespace std;

void train(vector<int>, const char *, const char *);
void test(vector<int>, const char *, const char *);
int findLabel(vector<float>);
int findLabelOutput(vector<float>);

int main(int argc, char* argv[])
{
    //input num of threads...

    //create topology of neural network
    int numLayers;
    int num;
    printf("input num of hidden layers: \n");
    scanf("%d", &numLayers);

    vector<int> topology;
    topology.push_back(784);

    for(int i=0; i<numLayers; ++i){
        printf("input num of neurons in layer[%d]: \n", i);
        scanf("%d", &num);
        topology.push_back(num);
    }
    topology.push_back(10);

    const char *weight_file = "weights.txt";
    const char *bias_file = "biases.txt";

    //training the neural network
    train(topology, bias_file, weight_file);

    //testing the neural network
    test(topology, bias_file, weight_file);

    return 0;
}

void train(vector<int> topology, const char * bias_file, const char * weight_file){

    //creating neural network
    NeuralNetwork nn(topology, 0.1);

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
                targetOutput.at(j) = (float)index;
            }else{
                fscanf(fp, "%d ", &index);
                targetInput.at(j-10) = (float)index/256;
            }
        }
        fclose(fp);

        //cout << "FEED FORWARD" << endl;
        nn.feedForword(targetInput);
        //cout << "BACK PROPAGATE " << endl;
        nn.backPropagate(targetOutput);
    }
    cout << "training complete\n";
    
    nn.print(bias_file, weight_file);
}

void test(vector<int> topology, const char * bias_file, const char * weight_file){

    NeuralNetwork nn(topology, 0.1, bias_file, weight_file);

    vector<float> targetOutput(10);
    //MNIST dataset pictures are 28x28
    vector<float> targetInput(784);
    int temp;
    string test;

    int correct_guess = 0;

    for(int index = 0; index < 10000; ++index){
        test = "/home/koshek/Desktop/MS/zadaci/zad1/test/test_";
        test.append(to_string(index));
        test.append(".txt");
        const char * ctest = test.c_str();
        
        FILE *fp = fopen(ctest, "r");
        assert(fp != NULL);
        
        for(int j = 0; j < 794; ++j){
            if(j<10){
                fscanf(fp, "%d ", &temp);
                targetOutput.at(j) = (float)temp;
            }else{
                fscanf(fp, "%d ", &temp);
                targetInput.at(j-10) = (float)temp/256;
            }
        }
        fclose(fp);


        nn.feedForword(targetInput);
        vector<float> preds = nn.getPredictions();

        int trained_label = findLabelOutput(preds);
        int actual_label = findLabel(targetOutput);

        if(trained_label == actual_label)
            correct_guess++;

        //cout << "should've gotten: " << actual_label << " actually got: " << trained_label << endl;
        
    }

    cout << "accuracy of simple neural network: " << correct_guess/100 << "%" << endl; 

}

int findLabel(vector<float> output){

    int label = 0;
    for(auto pt: output){
        if(pt == 0)
            label++;
        else
            break;
    }
    return label;
}

int findLabelOutput(vector<float> output){

    int label = 0;
    float max = output.at(label);

    for(int i = 0; i < output.size(); ++i){
        if(max < output.at(i)){
            label = i;
            max = output.at(i);
        }
    }
    return label;
}