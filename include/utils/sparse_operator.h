//
// Created by zjh on 2019/11/25.
//

#ifndef ADMM_END_SPARSE_OPERATOR_H
#define ADMM_END_SPARSE_OPERATOR_H

#include <omp.h>
#include "prob.h"

class sparse_operator{
public:
    static double nrm2_sq(const FeatureNode *x){
        double ret = 0;
        while(x->index != -1){
            ret += x->value*x->value;
            x++;
        }
        return (ret);
    }

    static double dot(const double *s, const FeatureNode *x){
        double ret = 0;
        while(x->index != -1){
            ret += s[x->index-1]*x->value;
            x++;
        }
        return (ret);
    }

    static void axpy(const double a, const FeatureNode *x, double *y){
        while(x->index != -1){
            y[x->index-1] += a*x->value;
            x++;
        }
    }

    static void axpy_omp(const double a, const FeatureNode *x, double *y, int nnz){
#pragma omp parallel for schedule(static)
        for(int k = 0; k < nnz; k++){
            const FeatureNode *xk = x + k;
            y[xk->index-1] += a*xk->value;
        }
    }
};

class Reduce_Vectors{
public:
    Reduce_Vectors(int size);
    ~Reduce_Vectors();

    void init(void);
    void sum_scale_x(double scalar, FeatureNode *x);
    void reduce_sum(double* v);

private:
    int nr_thread;
    int size;
    double **tmp_array;
};

#endif //ADMM_END_SPARSE_OPERATOR_H
