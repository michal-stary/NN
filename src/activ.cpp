#include "activ.h"
#include <algorithm>
#include <cmath>

void Relu::compute_outputs(matrix &eval_l) {
#pragma omp parallel for
    for (int k = 0; k < eval_l.height; k++){

        for (int l = 0; l < eval_l.width; l++){
            if(eval_l[k][l] < 0)
                eval_l[k][l] = 0;
        }
    }
}

// chace_l is initialized to target shape, eval_l is initialized to target shape
void Relu::compute_outputs(matrix &eval_l, matrix &cache_l) {
#pragma omp parallel for
    for (int k = 0; k < eval_l.height; k++){

        for (int l = 0; l < eval_l.width; l++){
            if(eval_l[k][l] < 0) {
                eval_l[k][l] = 0;
                cache_l[k][l] = 0;
            }
            else{
                cache_l[k][l] = 1;
            }
        }
    }
}

void Relu::compute_grads(const matrix &cache_l, matrix &dx_l) {
    dx_l *= cache_l;
}


void LogSoftmax::compute_outputs(matrix &eval_l) {

#pragma omp parallel for
    for (int i = 0; i < eval_l.height; i++){
        double sum = 0;
        double max = *std::max_element(eval_l[i].begin(), eval_l[i].end());

        for (int j = 0; j < eval_l.width; j++){
            eval_l[i][j] -= max;
            sum += std::exp(eval_l[i][j]);
        }
        double log_sum = std::log(sum);

        assert(!std::isnan(log_sum));

        for  (int j = 0; j < eval_l.width; j++){
            eval_l[i][j] -= log_sum;
        }
    }
}

void LogSoftmax::compute_outputs(matrix &eval_l, matrix &cache_l) {
    compute_outputs(eval_l);
}

void LogSoftmax::compute_grads(const matrix &cache_l, matrix &dx_l) {
    // skipped due to direct computation of d_loss w.r.t the logits

}

void Sigmoid::compute_outputs(matrix &eval_l) {
#pragma omp parallel for
    for (int i = 0; i < eval_l.height; i++) {

        for (int j = 0; j < eval_l.width; j++) {
            eval_l[i][j] = 1/(1+std::exp(-eval_l[i][j]));
        }
    }
}

void Sigmoid::compute_outputs(matrix &eval_l, matrix &cache_l) {
#pragma omp parallel for
    for (int i = 0; i < eval_l.height; i++) {

        for (int j = 0; j < eval_l.width; j++) {
            eval_l[i][j] = 1/(1+std::exp(-eval_l[i][j]));
            cache_l[i][j] = eval_l[i][j];
        }
    }
}

void Sigmoid::compute_grads(const matrix &cache_l, matrix &dx_l) {
#pragma omp parallel for
    for (int i = 0; i < dx_l.height; i++) {

        for (int j = 0; j < dx_l.width; j++) {
            dx_l[i][j] *= cache_l[i][j]*(1-cache_l[i][j]);
        }
    }
}

void Identity::compute_outputs(matrix &eval_l) {}

void Identity::compute_outputs(matrix &eval_l, matrix &cache_l) {
}

void Identity::compute_grads(const matrix &cache_l, matrix &dx_l) {
}
