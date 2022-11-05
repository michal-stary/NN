#include <cassert>
#include "net.h"
#include "optim.h"

matrix& NeuralNet::forward(const vecs& X, int start, int end, bool use_grads) {
    is_cached = use_grads;

    layers[0].compute_outputs(X, start, end, eval, cache, 0, use_grads);

    for(int l = 1; l<layers.size(); l++){
        layers[l].compute_outputs(eval, cache, l, use_grads);
    }


    matrix& log_pred = eval.back();
    return log_pred;
}

void NeuralNet::backward(const std::vector<int>& y, const vecs& X, int start, int end) {
    assert(is_cached);

    // steal the memory from the forward pass and reuse it
    comp& dx = eval;

    // calculate d_loss (pred - y)
    //dx.back() = pred;
    // dx.back() already contains pred

    dx.back().exp();

    for (int i = 0; i < dx.back().height; i++){
        dx.back()[i][y.at(start+i)] -= 1;
    }

    // mean reduction

    //dx.back() /= dx.back().height;

    // backward pass
    for (int l = layers.size() - 1; l > 0 ; l--){
        layers[l].compute_grads(eval, cache, dx, l);
    }
    layers[0].compute_grads(X, start, end, cache, dx, 0);
}

std::vector<int> NeuralNet::predict(vecs &X) {

    int n_batches = X.size() / batch_size;
    int old_size = X.size();
    // last incomplete batch is padded
    X.resize((n_batches + 1)*batch_size, std::vector<double>(X[0].size(), 0));

    std::vector<int> ord_preds(X.size());

    for (int b = 0; b < n_batches + 1 ; b++) {
        int start = b * batch_size;

        //int end = b != n_batches ? start + batch_size : X.size();
        int end = start + batch_size;

        // evaluate, no grads computed
        matrix& log_pred = forward(X, start, end, false);

        ordinal_argmax(log_pred, ord_preds, start);
    }

    // remove padding
    for (int i = 0; i < (n_batches + 1)*batch_size - old_size; i++){
        ord_preds.pop_back();
        X.pop_back();
    }

    return ord_preds;
}

void NeuralNet::add(Layer l) {
    layers.push_back(l);
    eval.push_back(matrix(batch_size, l.n_outputs));
    cache.push_back(matrix(batch_size, l.n_outputs));
}