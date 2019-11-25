//
// Created by zjh on 2019/11/25.
//

#ifndef ADMM_END_FUNCTION_H
#define ADMM_END_FUNCTION_H

class function{
public:
    virtual double fun(double *w,double *yi,double *zi) = 0 ;
    virtual void grad(double *w, double *g,double *yi,double *zi) = 0 ;

    virtual void Hv(double *s, double *Hs) = 0 ;
    virtual int get_nr_variable(void) = 0 ;
    virtual void get_diagH(double *M) = 0 ;

    virtual ~function(void){}
};

#endif //ADMM_END_FUNCTION_H
