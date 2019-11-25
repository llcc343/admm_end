//
// Created by zjh on 2019/11/25.
//

#ifndef ADMM_END_L2R_LR_FUN_MULTICORE_TRON_H
#define ADMM_END_L2R_LR_FUN_MULTICORE_TRON_H

#include <vector>
#include <cmath>
#include "function.h"
#include "prob.h"
#include "../utils/sparse_operator.h"

class l2r_lr_fun_multicore_tron : public function {
public:
    l2r_lr_fun_multicore_tron(const Problem *prob, double rho);
    ~l2r_lr_fun_multicore_tron();

    double fun(double *x,double *y_dual,double *z);
    void grad(double *w, double *g,double *yi,double *zi);
    void Hv(double *s, double *Hs);

    int get_nr_variable(void);
    void get_diagH(double *M);

    void batch_grad(double *w,double *y,double *z,double *g,std::vector<int> &batch_val);

private:
    void Xv(double *v, double *Xv);
    void XTv(double *v, double *XTv);

    double rho_;
    double *z;
    double *D;
    Reduce_Vectors *reduce_vectors;

    const Problem *prob;
};


#endif //ADMM_END_L2R_LR_FUN_MULTICORE_TRON_H
