//
// Created by zjh on 2019/11/25.
//

#ifndef ADMM_END_MATH_UTILS_H
#define ADMM_END_MATH_UTILS_H

#include <iostream>
#include <cmath>
#include "../sub_problem/function.h"

namespace math_utils{
    double norm_2(double *s,int dim);

    double dotProduct(const double *a, const double *b, int max_dim);

    bool lineSearch(double *x,double *y,double *z,double *g,double *d,
                    double *x_new,int max_dim, function *function,int max_search_num = 40,
                    double r = 0.5, double c = 1e-4);
}


#endif //ADMM_END_MATH_UTILS_H
