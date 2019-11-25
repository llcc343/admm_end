//
// Created by zjh on 2019/11/25.
//

#ifndef ADMM_END_TRON_H
#define ADMM_END_TRON_H

#include <cmath>
#include <cstring>
#include <stdio.h>
#include <stdarg.h>
#include "function.h"

class TRON {
public:
    TRON(const function *fun_obj, double eps = 0.1, double eps_cg = 0.1, int max_iter = 1000);
    ~TRON();

    void tron(double *w,double *yi,double *zi);
    void set_print_string(void (*i_print) (const char *buf));

private:
    int trpcg(double delta, double *g, double *M, double *s, double *r, bool *reach_boundary);
    double norm_inf(int n, double *x);

    double eps;
    double eps_cg;
    int max_iter;
    function *fun_obj;
    void info(const char *fmt,...);
    void (*tron_print_string)(const char *buf);
};


#endif //ADMM_END_TRON_H
