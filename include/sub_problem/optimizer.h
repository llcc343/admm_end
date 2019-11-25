//
// Created by zjh on 2019/11/25.
//

#ifndef ADMM_END_OPTIMIZER_H
#define ADMM_END_OPTIMIZER_H


class Optimizer {
public:
    virtual ~Optimizer(){}

    virtual void train(double *x,double *y,double *z)=0;
};


#endif //ADMM_END_OPTIMIZER_H
