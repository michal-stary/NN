#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "utils.h"
#include "init.h"
#include "activ.h"

class Layer{
    Activation* activ;

public:
    matrix weights;
    vec bias;

    matrix d_w;
    vec d_b;


    int n_inputs;
    int n_outputs;

    Layer(int n_inputs, int n_outputs, Activation* activ, Initializer* initializer):
            n_inputs(n_inputs),
            n_outputs(n_outputs),
            activ(activ),
            d_w(n_inputs, n_outputs),
            d_b(n_outputs){
        initializer->init(weights, n_inputs, n_outputs);
        initializer->init(bias, n_outputs);
    }


    void compute_outputs(comp& eval, comp& cache, int i, bool use_grads=true) {
        matmul(eval[i-1], weights, eval[i]);
        eval[i].add_vec(bias);

        if (use_grads)
            activ->compute_outputs(eval[i], cache[i]);
        else
            activ->compute_outputs(eval[i]);
    }

    void compute_outputs(const vecs& X, int start, int end, comp& eval, comp& cache, int i, bool use_grads=true) {
        matmul(X, start, end, weights, eval[i]);
        eval[i].add_vec(bias);
        //cache[i] = eval[i];
        if (use_grads)
            activ->compute_outputs(eval[i], cache[i]);
        else
            activ->compute_outputs(eval[i]);
    }



    void compute_grads(const comp& eval, const comp& cache, comp& dx, int i) {

        // zero grad call
        d_w.zero();
        d_b.zero();

        // compute grads w.r.t. w and b
        matmul_t1(eval[i-1], dx[i], d_w);
        dx[i].sum_to(d_b, 1);

        // compute grads w.r.t. dx
        matmul_t2(dx[i], weights, dx[i-1]);
        activ->compute_grads(cache[i-1], dx[i-1]);

    }
    void compute_grads(const vecs& X, int start, int end, const comp& cache, comp& dx, int i) {

        // zero grad call
        d_w.zero();
        d_b.zero();

        // compute grads w.r.t. w and b
        matmul_t1(X, start, end, dx[i], d_w);
        dx[i].sum_to(d_b, 1);
    }


};


#endif