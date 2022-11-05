#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <tuple>
#include <algorithm>
#include <unordered_set>

using vecs = std::vector<std::vector<double>>;


class Loader{
    const std::string& dir_path;
public:
    Loader(const std::string& dir_path): dir_path(dir_path){}

    std::vector<int> read_labels(const std::string& file);
    vecs read_X(const std::string&  file);
    std::unordered_set<int> get_indices(int len, int ratio);

    template<typename T>
    std::tuple<T,T> split(const T& array, const std::unordered_set<int>& indices){
        T train;
        T val;
        for (int i= 0; i<array.size(); i++){
            if (indices.find(i) != indices.end()){
                train.push_back(array[i]);
            }
            else{
                val.push_back(array[i]);
            }
        }
        return {train, val};
    }
};


/*
class Generator_train_iter{
    const vecs& X;
    const vecs& y;
    int start;
    int batch_size;
    int size;
    bool finished = false;
public:
    Generator_train_iter(){}

    Generator_train_iter &operator++()
    {
        start += batch_size;
        if (start > size){
            finished = true;
        }
    }

    bool operator!=( const Generator_train_iter &o ) const
    {
        return start != o.start || size != o.size || batch_size != o.batch_size;
    }

    std::tuple<vecs, std::vector<int>> operator*(){
        return std::make_tuple();
    }
    int  operator*() const { return node->value; }

};

class Generator_test_iter{
    vecs::iterator it_X;
    int batch_size;
public:
    Generator_test_iter();
    Generator_test_iter &operator++();

    bool operator!=( const Generator_iter &o ) const;

    int &operator*();
    int  operator*() const;

};

class Generator{
    int batch_size;
public:
    Generator():
            it(Generator_iter(X, y, batch_size)){
        gen_y = true;
    }

    Generator(const vecs& X, int batch_size):
            it(Generator_iter(X, batch_size)){
        gen_y = false;
    }
};


class Generator_train: public Generator{
public:
    Generator_train(const vecs& X, const std::vector<int>& y, int batch_size);
};

class Generator_test: public Generator{
public:
    Generator_test(const vecs& X, int batch_size);
};
*/
