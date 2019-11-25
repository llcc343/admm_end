//
// Created by zjh on 2019/11/25.
//

#include "sub_problem/l2r_lr_fun_multicore_tron.h"


l2r_lr_fun_multicore_tron::l2r_lr_fun_multicore_tron(const Problem *prob, double rho){
    int l=prob->l;

    this->prob = prob;

    z = new double[l];
    D = new double[l];

    reduce_vectors = new Reduce_Vectors(get_nr_variable());

    rho_=rho;
}

l2r_lr_fun_multicore_tron::~l2r_lr_fun_multicore_tron(){
    delete[] z;
    delete[] D;
    delete reduce_vectors;
}

//目标函数
double l2r_lr_fun_multicore_tron::fun(double *w,double *yi,double *zi){
    int i;
    double f=0;
    double *y=prob->y;
    int l=prob->l;
    int w_size=get_nr_variable();

    Xv(w, z);//权重与数据集的乘积，得到的结果保存在z中

#pragma omp parallel for private(i) reduction(+:f) schedule(static)
    for(i=0;i<l;i++)
    {
        double yz = y[i]*z[i];
        if (yz >= 0)
            f += log(1 + exp(-yz));
        else
            f += (-yz+log(1 + exp(yz)));
    }

// #pragma omp parallel for private(i) reduction(+:f) schedule(static)
    for (i = 0; i < w_size; ++i) {
        f+=yi[i]*(w[i]-zi[i])+0.5*rho_*(w[i]-zi[i])*(w[i]-zi[i]);
    }

    return f;
}

//梯度
void l2r_lr_fun_multicore_tron::grad(double *w, double *g,double *yi,double *zi){
    int i;
    double *y=prob->y;
    int l=prob->l;
    int w_size=get_nr_variable();

#pragma omp parallel for private(i) schedule(static)
    for(i=0;i<l;i++){
        z[i] = 1/(1 + exp(-y[i]*z[i]));
        D[i] = z[i]*(1-z[i]);
        z[i] = (z[i]-1)*y[i];
    }
    XTv(z, g);

// #pragma omp parallel for schedule(static)
    for(i=0;i<w_size;++i)
        g[i]+=rho_ *(w[i]-zi[i]) + yi[i];
}

int l2r_lr_fun_multicore_tron::get_nr_variable(void){
    return prob->n;
}

void l2r_lr_fun_multicore_tron::get_diagH(double *M){
    int i;
    int l = prob->l;
    int w_size=get_nr_variable();
    FeatureNode **x = prob->x;

    for (i=0; i<w_size; i++)
        M[i] = 1;

    for (i=0; i<l; i++){
        FeatureNode *s = x[i];
        while (s->index!=-1){
            M[s->index-1] += s->value*s->value*D[i];
            s++;
        }
    }
}

void l2r_lr_fun_multicore_tron::Hv(double *s, double *Hs){
    int i;
    int l=prob->l;
    int w_size=get_nr_variable();
    FeatureNode **x=prob->x;

    reduce_vectors->init();

#pragma omp parallel for private(i) schedule(guided)
    for(i=0;i<l;i++){
        FeatureNode * const xi=x[i];
        double xTs = sparse_operator::dot(s, xi);

        xTs = D[i]*xTs;

        reduce_vectors->sum_scale_x(xTs, xi);
    }

    reduce_vectors->reduce_sum(Hs);
#pragma omp parallel for private(i) schedule(static)
    for(i=0;i<w_size;i++)
        Hs[i] = s[i] + Hs[i];
}

//将向量v与数据集相乘，得到向量放入Vx中。也就是权重乘以数据集
void l2r_lr_fun_multicore_tron::Xv(double *v, double *Xv){
    int i;
    int l=prob->l;
    FeatureNode **x=prob->x;

#pragma omp parallel for private (i) schedule(guided)
    for(i=0;i<l;i++)
        Xv[i]=sparse_operator::dot(v, x[i]);
}

//将一个元素与对应的数据集一行相乘，然后返回到XTv中，XTv累加起来
void l2r_lr_fun_multicore_tron::XTv(double *v, double *XTv){
    int i;
    int l=prob->l;
    FeatureNode **x=prob->x;

    reduce_vectors->init();

#pragma omp parallel for private(i) schedule(guided)
    for(i=0;i<l;i++)
        reduce_vectors->sum_scale_x(v[i], x[i]);

    reduce_vectors->reduce_sum(XTv);
}