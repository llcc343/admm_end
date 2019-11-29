//
// Created by zjh on 2019/11/25.
//

#ifndef ADMM_END_L2R_LR_FUN_H
#define ADMM_END_L2R_LR_FUN_H

#include <cmath>
#include "function.h"
#include "prob.h"
#include "../utils/sparse_operator.h"

class l2r_lr_fun : public function{
public:
    l2r_lr_fun(const Problem *prob, double rho);
    ~l2r_lr_fun();

    double fun(double *w,double *y, double *z);
    void grad(double *w, double *g,double *y, double *z);
    void Hv(double *s, double *Hs);

    int get_nr_variable();
    void get_diagH(double *M);

private:
    void Xv(double *v, double *Xv);
    void XTv(double *v, double *XTv);

    double rho_;
    double *z;
    double *D;
    const Problem *prob_;
};


#endif //ADMM_END_L2R_LR_FUN_H
