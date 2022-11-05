#ifndef NN_NET_H
#define NN_NET_H

#include "utils.h"
#include "activ.h"
#include "layer.h"
#include "optim.h"


using vecs = std::vector<std::vector<double>>;



class NeuralNet{
    int input_size;
    int output_size;
    int batch_size;

    bool is_cached = false;
    comp eval;
    comp cache;

    friend class Trainer;

public:
    std::vector<Layer> layers;


    NeuralNet(int batch_size, int input_size, int output_size):
            batch_size(batch_size),
            input_size(input_size),
            output_size(output_size){}

    matrix& forward(const vecs& X, int start, int end, bool use_grads);
    void backward(const std::vector<int>& y, const vecs& X, int start, int end);


    std::vector<int> predict(vecs &X);

    void add(Layer l);

};



#endif