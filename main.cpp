#include <iostream>
#include "src/data.h"
#include "src/optim.h"
#include "src/net.h"
#include "src/utils.h"
#include "src/trainer.h"
#include <string>
int main() {

    std::string dir = "/home/michal/CLionProjects/NN/pv021_project/data/";
    int batch_size = 16;
    int n_inputs = 784;
    int n_outputs = 10;

    Loader loader(dir);

    auto X = loader.read_X("fashion_mnist_train_vectors.csv");
    auto y = loader.read_labels("fashion_mnist_train_labels.csv");
    auto train_indices = loader.get_indices(X.size(), 80);
    auto [X_train, X_val] = loader.split(X, train_indices);
    auto [y_train, y_val] = loader.split(y, train_indices);

    Ginit relu_init;
    Ginit lsm_init;
    Ginit sig_init;

    NeuralNet net(batch_size, n_inputs, n_outputs);

    Relu relu;
    LogSoftmax log_softmax;
//    Sigmoid sigmoid;
//    Identity id;

    net.add(Layer(n_inputs,256, &relu, &sig_init));
    net.add(Layer(256,100, &relu, &sig_init));
    net.add(Layer(100,n_outputs, &log_softmax, &lsm_init));

    Adam optimizer(net.layers, 0.001, 0.8, 0.99, false);

    Trainer trainer(net, optimizer);
    trainer.train(X_train, y_train, X_val, y_val, 5, 1);


    auto ord_pred_train = net.predict(X_train);
    double train_acc = accuracy_score(ord_pred_train, y_train);
    std::printf("Final training accuracy %.3f\n", train_acc);

    auto X_test = loader.read_X("fashion_mnist_test_vectors.csv");
    auto y_test = loader.read_labels("fashion_mnist_test_labels.csv");

    auto ord_pred_test = net.predict(X_test);
    double test_acc = accuracy_score(ord_pred_test, y_test);
    std::printf("Final test accuracy %.3f\n", test_acc);

    return 0;
}
