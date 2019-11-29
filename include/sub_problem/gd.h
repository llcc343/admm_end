//
// Created by zjh on 2019/11/25.
//

#ifndef ADMM_END_GD_H
#define ADMM_END_GD_H

#include "function.h"
#include "prob.h"
#include "optimizer.h"
#include "../utils/math_utils.h"

class Gd : public Optimizer {
public:
    Gd(const function *fun_obj,double eps=0.01,int max_iter=100);
    ~Gd();

    void train(double *x,double *y,double *z);

private:
    function *fun_obj_;
    double *g,*d,*x_new;
    double f,f_old;
    int max_iter_;
    int feature_num;
    double eps_;
};


#endif //ADMM_END_GD_H
