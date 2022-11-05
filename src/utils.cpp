#include "utils.h"
#include <cassert>
#include <cmath>
#include <omp.h>
#include <algorithm>

double accuracy_score(const std::vector<int>& ord_pred, const std::vector<int>& y){
    assert(ord_pred.size() == y.size());
    float correct = 0;
    for(int i = 0; i < y.size(); i++){
        if(ord_pred[i] == y[i])
            correct += 1;
    }
    return correct/y.size();
}




matrix::matrix(vecs data): data(data), height(data.size()), width(data[0].size()) {}

matrix::matrix(int k, int l):
        data(k,std::vector<double>(l, 0)),
        height(k),
        width(l)
        {}

vec::vec(int k):
    data(k,0),
    size(k)
    {}

void matmul(const vecs& data, int start, int end, const matrix& m2, matrix& res){
    assert(data[start].size() == m2.height);
    assert((end-start) == res.height);
    assert(m2.width == res.width);

#pragma omp parallel for
    for (int i = start; i < end; i++){

        for (int j = 0; j < m2.width; j++){
            res.data[i-start][j] = 0;
            for (int k = 0; k < data[start].size(); k++){
                res.data[i-start][j] += data[i][k] * m2.data[k][j];
                }
            }
        }
}


// m1 has (k, n) shape, m2 has (n, l) shape, res has (k, l) shape
void matmul(const matrix& m1, const matrix& m2, matrix& res){
    assert(m1.width == m2.height);
    assert(m1.height == res.height);
    assert(m2.width == res.width);
#pragma omp parallel for
    for (int i = 0; i < m1.height; i++){

        for (int j = 0; j < m2.width; j++){
            res.data[i][j] = 0;
            for (int k = 0; k < m1.width; k++){
                res.data[i][j] += m1.data[i][k] * m2.data[k][j];
            }
        }
    }
}

void matmul_t1(const matrix& m1, const matrix& m2, matrix& res){
    assert(m1.height == m2.height);
    assert(m1.width == res.height);
    assert(m2.width == res.width);
#pragma omp parallel for
    for (int i = 0; i < m1.width; i++){

        for (int j = 0; j < m2.width; j++){
            res.data[i][j] = 0;
            for (int k = 0; k < m1.height; k++){
                res.data[i][j] += m1.data[k][i] * m2.data[k][j];
            }
        }
    }
}

void matmul_t1(const vecs& data, int start, int end, const matrix& m2, matrix& res){
    assert((end-start) == m2.height);
    assert(data[start].size() == res.height);
    assert(m2.width == res.width);

#pragma omp parallel for
    for (int i = 0; i < data[start].size(); i++){

        for (int j = 0; j < m2.width; j++){
            res.data[i][j] = 0;
            for (int k = start; k < end; k++){
                res.data[i][j] += data[k][i] * m2.data[k-start][j];
            }
        }
    }
}


void matmul_t2(const matrix& m1, const matrix& m2, matrix& res){
    assert(m1.width == m2.width);
    assert(m1.height == res.height);
    assert(m2.height == res.width);
#pragma omp parallel for
    for (int i = 0; i < m1.height; i++){

        for (int j = 0; j < m2.height; j++){
            res.data[i][j] = 0;
            for (int k = 0; k < m1.width; k++){
                res.data[i][j] += m1.data[i][k] * m2.data[j][k];
            }
        }
    }
}


// m1 has (k, n) shape, m2 has (n, l) shape
matrix matmul(const matrix& m1, const matrix& m2){
    assert(m1.width == m2.height);
    matrix res(m1.height, m2.width);
    matmul(m1, m2, res);
    return res;
}

void matrix::operator+=(const matrix& o) {
#pragma omp parallel for
    for (int k = 0; k < height; k++) {

        for (int l = 0; l < width; l++) {
            data[k][l] += o[k][l];
        }
    }
}


void matrix::operator-=(const matrix& o) {
    #pragma omp parallel for
    for (int k = 0; k < height; k++) {

        for (int l = 0; l < width; l++) {
            data[k][l] -= o[k][l];
        }
    }
}

void matrix::operator*=(const matrix& o){
#pragma omp parallel for
    for (int k = 0; k < height; k++) {

        for (int l = 0; l < width; l++) {
            data[k][l] *= o[k][l];
        }
    }
}


void matrix::operator*=(double o) {
    #pragma omp parallel for
    for (int k = 0; k < height; k++) {

        for (int l = 0; l < width; l++) {
            data[k][l] *= o;
        }
    }
}
void matrix::operator/=(double o) {
#pragma omp parallel for
    for (int k = 0; k < height; k++) {

        for (int l = 0; l < width; l++) {
            data[k][l] /= o;
        }
    }
}


matrix operator*(double a, matrix r){
#pragma omp parallel for
    for (int k = 0; k < r.height; k++) {

        for (int l = 0; l < r.width; l++) {
            r.data[k][l] *= a;
        }
    }
    return r;
}
vec operator*(double a, vec r){
#pragma omp parallel for
    for (int k = 0; k < r.size; k++) {
        r.data[k] *= a;
    }
    return r;
}

vec operator/(vec l, const vec& r){
    assert(l.size == r.size);
    #pragma omp parallel for
    for (int k = 0; k < r.size; k++) {
        l.data[k] /= r.data[k];
    }
    return r;
}

void vec::operator*=(double o) {
    #pragma omp parallel for
    for (int k = 0; k < size; k++) {
        data[k] *= o;
    }
}

void vec::operator-=(const vec& o) {
    #pragma omp parallel for
    for (int k = 0; k < size; k++) {
        data[k] -= o.data[k];
    }
}

void vec::operator+=(const vec& o) {
#pragma omp parallel for
    for (int k = 0; k < size; k++) {
        data[k] += o.data[k];
    }
}

void matrix::add_vec(const vec& v){
    #pragma omp parallel for
    for (int k = 0; k < height; k++) {

        for (int l = 0; l < width; l++) {
            data[k][l] += v.data[l];
        }
    }
}


matrix& matrix::exp(){
#pragma omp parallel for
    for (int k = 0; k < height; k++) {

        for (int l = 0; l < width; l++) {
            data[k][l] = std::exp(data[k][l]);
        }
    }
    return *this;
}


void matrix::sum_to(vec& res, int axis){
#pragma omp parallel for
    for (int k = 0; k < width; k++) {

        for (int l = 0; l < height; l++) {
            res.data[k] += data[l][k];
        }
    }
}

void matrix::zero() {
#pragma omp parallel for
    for (int k = 0; k < height; k++) {

        for (int l = 0; l < width; l++) {
            data[k][l] = 0;
        }
    }
}

void vec::zero() {
#pragma omp parallel for
    for (int k = 0; k < size; k++) {
        data[k] = 0;
    }
}

void matrix::pow2(){
#pragma omp parallel for
    for (int k = 0; k < height; k++) {

        for (int l = 0; l < width; l++) {
            data[k][l] = pow(data[k][l], 2);
        }
    }
}

matrix sqrt_eps(matrix m, double eps){
#pragma omp parallel for
    for (int k = 0; k < m.height; k++) {

        for (int l = 0; l < m.width; l++) {
            m.data[k][l] = sqrt(m.data[k][l]) + eps;
        }
    }
    return m;
}


void vec::pow2(){
#pragma omp parallel for
    for (int k = 0; k < size; k++) {
        data[k] = pow(data[k], 2);
    }
}

vec sqrt_eps(vec v, double eps){
#pragma omp parallel for
    for (int k = 0; k < v.size; k++) {
        v.data[k] = sqrt(v.data[k]) + eps;
    }
    return v;
}

matrix operator/(matrix l, const matrix& r){
    assert(l.height == r.height && l.width == r.width);
#pragma omp parallel for
    for(int i = 0; i < l.height; i++){

        for(int j = 0; j < l.width; j++){
            l[i][j] /= r[i][j];
        }
    }
    return l;
}

matrix operator+(matrix l, const matrix& r){
    assert(l.height == r.height && l.width == r.width);
#pragma omp parallel for
    for(int i = 0; i < l.height; i++){

        for(int j = 0; j < l.width; j++){
            l[i][j] += r[i][j];
        }
    }
    return l;
}


matrix operator-(matrix l, const matrix& r){
    assert(l.height == r.height && l.width == r.width);
#pragma omp parallel for
    for(int i = 0; i < l.height; i++){

        for(int j = 0; j < l.width; j++){
            l[i][j] -= r[i][j];
        }
    }
    return l;
}


matrix operator*(matrix l, const matrix& r){
    assert(l.height == r.height && l.width == r.width);
#pragma omp parallel for
    for(int i = 0; i < l.height; i++){

        for(int j = 0; j < l.width; j++){
            l[i][j] *= r[i][j];
        }
    }
    return l;
}


/*

vector matrix::col(int n) const{
    vector col;
    for(const vector& row: data){
        col.push_back(row[n]);
    }
    return col;
}

vector operator*(const matrix& m, const vector& v){
    if (static_cast<int>(v.size()) != m.width())
        throw std::runtime_error("invalid dimensions");
    vector res;
    for(int i = 0; i < m.height(); i++){
        vector row = m.row(i);
        res.push_back(row*v);
    }
    return res;
}
vector operator*(const vector& v, const matrix& m){
    if (static_cast<int>(v.size()) != m.height())
        throw std::runtime_error("invalid dimensions");
    vector res;
    for(int i = 0; i < m.width(); i++){
        vector col = m.col(i);
        res.push_back(col*v);
    }
    return res;
}

*/




void ordinal_argmax(const matrix& log_pred, std::vector<int>& ord_preds, int start){
#pragma omp parallel for
    for (int i = 0; i < log_pred.height; i++){
        int max_i = 0;
        double max_val = log_pred[i][0];

        for(int j = 1; j < log_pred.width; j++){
            if (log_pred[i][j] > max_val){
                max_i = j;
                max_val = log_pred[i][j];
            }
        }
        ord_preds[start+i] = max_i;
    }
};
