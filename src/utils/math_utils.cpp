//
// Created by zjh on 2019/11/25.
//

#include "utils/math_utils.h"

namespace math_utils{
    double norm_2(double *s,int dim){
        double norm_2_val=0;
        for(int i = 0; i < dim; i++){
            norm_2_val+=s[i]*s[i];
        }
        norm_2_val=sqrt(norm_2_val);
        return norm_2_val;
    }

    double dotProduct(const double *a, const double *b, int max_dim) {
        double temp = 0.0;
        for (int i = 0; i < max_dim; ++i) {
            temp += a[i] * b[i];
        }
        return temp;
    }

    bool lineSearch(double *x,double *y,double *z, double *g, double *d,double *x_new,int max_dim,
                    function *fun_obj,int max_search_num, double r, double c) {
        int k = 0;
        double func_val, func_val_next;
        double step = 1.0;

        func_val = fun_obj->fun(x, y, z);
        double dginit = dotProduct(g, d, max_dim);
        if (dginit > 0) {
            std::cerr << "dginit > 0" << std::endl;
            exit(-1);
        }
        while (k < max_search_num) {
            ++k;

            for (int i = 0; i < max_dim; ++i) {
                x_new[i] = x[i] + step * d[i];
            }
            func_val_next = fun_obj->fun(x_new, y, z);

            if (func_val_next <= func_val + c * step * dginit) {
                return true;
            } else {
                step *= r;
                if (!std::isfinite(step)) {
                    std::cerr << "线性搜索步长失败" << std::endl;
                    exit(-1);
                }
            }
        }

        return false;
    }
}