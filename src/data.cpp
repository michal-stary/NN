#include <vector>
#include <string>
#include <unordered_set>
#include <random>

#include "data.h"

using vecs = std::vector<std::vector<double>>;


std::vector<int> Loader::read_labels(const std::string& file){
    std::vector<int> labels;
    auto d = dir_path + file;
    std::ifstream f( d);
    std::string line;

    while( std::getline(f, line)){

        labels.push_back(std::stoi(line));
    }
    return labels;
};

vecs Loader::read_X(const std::string&  file){
    vecs X;
    std::ifstream f( dir_path + file );


    std::string line;
    double val;
    while( std::getline(f, line)){

        // Create a stringstream of the current line
        std::stringstream ss(line);

        std::vector<double> sample;

        // Extract each integer
        while(ss >> val) {

            // Add the current integer to the 'colIdx' column's values vector
            sample.push_back(val/255-0.5);

            // If the next token is a comma, ignore it and move on
            if (ss.peek() == ',') ss.ignore();

        }
        X.push_back(sample);
    }
    return X;
}

std::unordered_set<int> Loader::get_indices(int len, int ratio) {
    // inspired from https://stackoverflow.com/questions/28287138/c-randomly-sample-k-numbers-from-range-0n-1-n-k-without-replacement
    std::mt19937 gen;
    int k = len*ratio/100;
    std::unordered_set<int> elems;
    for (int r = len - k; r < len; ++r) {
        int v = std::uniform_int_distribution<>(0, r)(gen);
//        int v = 0;
        // there are two cases.
        // v is not in candidates ==> add it
        // v is in candidates ==> well, r is definitely not, because
        // this is the first iteration in the loop that we could've
        // picked something that big.
        if (!elems.insert(v).second) {
            elems.insert(r);
        }
    }
    return elems;
}

//
//std::tuple<std::vector<int>,std::vector<int>> Loader::split_labels(const std::vector<int>& labels, const std::unordered_set<int>& indices){
//    vecs X_train;
//    vecs X_val;
//    for (int i= 0; i<X.size(); i++){
//        if (indices.find(i) != indices.end()){
//            X_train.push_back(X[i]);
//        }
//        else{
//            X_val.push_back(X[i]);
//        }
//    }
//    return {X_train, X_val};
//}
