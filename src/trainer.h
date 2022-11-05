//
// Created by michal on 06.01.21.
//

#ifndef NN_TRAINER_H
#define NN_TRAINER_H
#include "net.h"
#include "optim.h"
#include <chrono>

class Trainer{
    NeuralNet& model;
    Optimizer& optim;
public:
    Trainer(NeuralNet& model, Optimizer& optimizer):
        model(model),
        optim(optimizer){}

    void train(vecs& X_train, const std::vector<int>& y_train,
               vecs& X_val, const std::vector<int>& y_val, int n_epochs, int verbose=0){
        int n_batches = static_cast<int>(X_train.size()) / model.batch_size;
        double total_train_time = 0;

        for(int e = 0; e < n_epochs; e++) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            double epoch_loss = 0 ;

            // last incomplete batch is skipped (shrug)
            for (int b = 0; b < n_batches ; b++) {
                int start = b * model.batch_size;

                //int end = b != n_batches ? start + batch_size : X.size();
                int end = start + model.batch_size;

                // evaluate and calculate grads
                matrix& log_pred = model.forward(X_train, start, end, true);

                // calculate loss
                double loss_batch = 0;
                for (int i = 0; i < log_pred.height; i++){
                    loss_batch += -log_pred[i][y_train[start+i]];
                }

                model.backward(y_train, X_train, start, end);

                // update weights
                optim.step();

                epoch_loss += loss_batch;

                if (verbose>=2) {
                    std::printf("Batch %d / %d done\n", b+1, n_batches);
                }

                if (verbose >= 3)
                    std::printf("Batch loss was %.5f\n", loss_batch);
            }
            auto epoch_finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> epoch_elapsed = epoch_finish - epoch_start;
            epoch_loss /= y_train.size();
            total_train_time += epoch_elapsed.count();
            if (verbose>=1) {
                std::printf("Epoch %d / %d done\n", e+1, n_epochs);
                std::printf("Epoch loss was %.5f\n", epoch_loss);
                std::printf("Epoch took %.1f seconds to train\n", epoch_elapsed.count());

            }
            auto ord_pred_train = model.predict(X_train);
            auto ord_pred_val = model.predict(X_val);

            double train_acc = accuracy_score(ord_pred_train, y_train);
            double val_acc = accuracy_score(ord_pred_val, y_val);

            std::printf("\nTraining accuracy after %d epochs %.3f\n", e+1, train_acc);
            std::printf("Validation accuracy after %d epochs %.3f\n", e+1, val_acc);
            std::printf("\nTraining took %.1f seconds so far\n\n", total_train_time);
        }
    }
};



#endif //NN_TRAINER_H
