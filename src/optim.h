//#include "utils.h"


#ifndef NN_OPTIM_H
#define NN_OPTIM_H

#include "layer.h"


class Optimizer{
public:
    double lr;
    std::vector<Layer>& parameters;
    Optimizer(std::vector<Layer>& parameters, double lr):
        lr(lr),
        parameters(parameters){}

    virtual void step()=0;

};


class SGD: public Optimizer{
    std::vector<matrix> state_w;
    std::vector<vec> state_b;
    double mu;
public:
    SGD(std::vector<Layer>& parameters, double lr, double mu):
        Optimizer(parameters, lr),
        mu(mu){
        for (auto& p: parameters){
            state_w.emplace_back(p.weights.height, p.weights.width);
            state_b.emplace_back(p.bias.size);
        }
    }

    void step() override;

};

class Adam: public Optimizer{
    std::vector<matrix> state1_w, state2_w;
    std::vector<vec> state1_b, state2_b;
    double beta1, beta2;
    double eps = 0.000001;
    int t=0;
    bool bias_correction;

public:
    Adam(std::vector<Layer>& parameters, double lr, double beta1, double beta2, bool bias_correction=false):
        Optimizer(parameters, lr),
        beta1(beta1),
        beta2(beta2),
        bias_correction(bias_correction)
        {
        for (auto& p: parameters){

            state1_w.emplace_back(p.weights.height, p.weights.width);
            state1_b.emplace_back(p.bias.size);

            state2_w.emplace_back(p.weights.height, p.weights.width);
            state2_b.emplace_back(p.bias.size);
        }
    }

    void step() override;

};

#endif