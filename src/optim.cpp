#include "optim.h"
#include <omp.h>
void SGD::step() {
    #pragma omp parallel for
    for (int p = 0; p < parameters.size(); p++){
        state_w[p] *= mu;
        state_b[p] *= mu;

        parameters[p].d_w *= 1-mu;
        parameters[p].d_b *= 1-mu;

        state_w[p] += parameters[p].d_w;
        state_b[p] += parameters[p].d_b;

        parameters[p].weights -= lr * state_w[p];
        parameters[p].bias -= lr * state_b[p];
    }
}

void Adam::step() {
#pragma omp parallel for
    for(int p = 0; p < parameters.size(); p++){
        state1_w[p] *= beta1;
        state1_b[p] *= beta1;

        state1_w[p] += (1-beta1) * parameters[p].d_w;
        state1_b[p] += (1-beta1) * parameters[p].d_b;

        state2_w[p] *= beta2;
        state2_b[p] *= beta2;

        parameters[p].d_w.pow2();
        parameters[p].d_b.pow2();

        parameters[p].d_w *= 1-beta2;
        parameters[p].d_b *= 1-beta2;


        state2_w[p] += parameters[p].d_w;
        state2_b[p] += parameters[p].d_b;

        if(bias_correction){
            t +=1;
            assert(false);
        }
        else{
            parameters[p].weights -= lr * (state1_w[p]/(sqrt_eps(state2_w[p], eps)));
            parameters[p].bias -= lr * lr * (state1_b[p]/(sqrt_eps(state2_b[p], eps)));
        }



    }
}