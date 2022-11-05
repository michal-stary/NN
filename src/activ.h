#ifndef NN_ACTIV_H
#define NN_ACTIV_H

#include "utils.h"


class Activation{
public:

    virtual void compute_outputs(matrix& eval_l, matrix& cache_l) = 0;
    virtual void compute_outputs(matrix& eval_l) = 0;
    virtual void compute_grads(const matrix& cache_l, matrix& dx_l) = 0;
    virtual ~Activation()=default;

};


class Relu: public Activation{
public:

    void compute_outputs(matrix& eval_l, matrix& cache_l) override;
    void compute_outputs(matrix& eval_l) override;
    void compute_grads(const matrix& cache_l, matrix& dx_ll) override;


};

class LogSoftmax: public Activation{
public:

    void compute_outputs(matrix& eval_l, matrix& cache_l) override;
    void compute_outputs(matrix& eval_l) override;
    void compute_grads(const matrix& cache_l, matrix& dx_l) override;

};

class Sigmoid: public Activation{
public:

    void compute_outputs(matrix& eval_l, matrix& cache_l) override;
    void compute_outputs(matrix& eval_l) override;
    void compute_grads(const matrix& cache_l, matrix& dx_l) override;


};

class Identity: public Activation{
public:

    void compute_outputs(matrix& eval_l, matrix& cache_l) override;
    void compute_outputs(matrix& eval_l) override;
    void compute_grads(const matrix& cache_l, matrix& dx_l) override;


};


#endif

