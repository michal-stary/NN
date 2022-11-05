#ifndef NN_INIT_H
#define NN_INIT_H

#include <random>
#include <algorithm>

class Initializer{
public:
    virtual void init(matrix& m, int n_inputs, int n_outputs)=0;
    virtual void init(vec& v, int n_outputs)=0;
    virtual ~Initializer()=default;
};

class Ginit: public Initializer{
public:
    void init(matrix& m, int n_inputs, int n_outputs) override{
        static std::default_random_engine generator(42);
        static std::normal_distribution<double> distribution(0,0.01);
        for (int i = 0; i < n_inputs; i++){
            std::vector<double> row(n_outputs);
            std::generate(row.begin(), row.end(), []() { return distribution(generator);});
            m.data.push_back(row);
        }
        m.height = n_inputs;
        m.width = n_outputs;
     }
    void init(vec& v, int n_outputs) override{
        v.data.resize(n_outputs, 0);
        v.size = n_outputs;
    }
};


#endif //NN_INIT_H
