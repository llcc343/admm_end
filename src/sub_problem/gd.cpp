//
// Created by zjh on 2019/11/25.
//

#include "sub_problem/gd.h"


Gd::Gd(const function *fun_obj, Problem *prob,double eps,int max_iter){
    fun_obj_=const_cast<function *>(fun_obj);
    feature_num=fun_obj_->get_nr_variable();
    g=new double[feature_num]();
    d=new double[feature_num]();
    x_new=new double[feature_num]();
    eps_=eps;
    max_iter_=max_iter;
}

Gd::~Gd(){
    delete []g;
    delete []d;
    delete []x_new;
}

void Gd::train(double *x,double *y,double *z){
    f=fun_obj_->fun(x,y,z);
    fun_obj_->grad(x,g,y,z);
    int k=0;
    while (k<max_iter_){
        for(int i=0;i<feature_num;i++){
            d[i]=-g[i];
        }
        if(math_utils::lineSearch(x,y,z,g,d,x_new,feature_num,fun_obj_)){
            for (int i = 0; i < feature_num; i++) {
                x[i]=x_new[i];
            }
            f_old=f;
            f=fun_obj_->fun(x,y,z);
            fun_obj_->grad(x,g,y,z);
            if(fabs(f-f_old)<eps_){
                break;
            }
        }else{
            std::cerr<<"line search fail!!!"<<std::endl;
        }
        k++;
    }
}