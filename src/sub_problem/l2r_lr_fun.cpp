//
// Created by zjh on 2019/11/25.
//

#include "sub_problem/l2r_lr_fun.h"


l2r_lr_fun::l2r_lr_fun(const Problem *prob, double rho):prob_(prob),rho_(rho){
    int l=prob->l;//dataNum

    z = new double[l];
    D = new double[l];
}

l2r_lr_fun::~l2r_lr_fun(){
    delete[] z;
    delete[] D;
}

//计算子问题函数值，w表示局部便利那个，yi表示对偶变量，zi表示全局变量
double l2r_lr_fun::fun(double *w,double *yi, double *zi){
    int i;
    double f=0;
    double *y=prob_->y;//label
    int l=prob_->l;//dataNum
    int w_size=get_nr_variable();//featureNum

    Xv(w, z);//权重和数据集的乘积，得到的值保存在z中。

    //计算局部变量
    for(i=0;i<w_size;i++)
        f += 0.5 * rho_*(w[i] - zi[i])*(w[i]-zi[i]) + yi[i]*(w[i]-zi[i]);
    //f /= 2.0;
    for(i=0;i<l;i++)
    {
        if (y[i] > 0)
            f += log(1 + exp(-z[i]));
        else
            f += (z[i]+log(1 + exp(-z[i])));
    }
    return(f);
}

//计算局部变量函数的梯度，w表示局部变量，g表示得到的梯度值，yi表示对偶变量，zi表示全局变量
void l2r_lr_fun::grad(double *w, double *g,double *yi, double *zi){
    int i;
    double *y=prob_->y;
    int l=prob_->l;
    int w_size=get_nr_variable();

    for(i=0;i<l;i++){
        z[i] = 1/(1 + exp(-y[i]*z[i]));
        D[i] = z[i]*(1-z[i]);
        z[i] = (z[i]-1)*y[i];
    }
    XTv(z, g);//得到g存储了lr的梯度

    for(i=0;i<w_size;i++)
        g[i] = g[i] + rho_ *(w[i]-zi[i]) + yi[i];
}

//获得特征个数
int l2r_lr_fun::get_nr_variable(void){
    return prob_->n;
}

void l2r_lr_fun::Hv(double *s, double *Hs){
    int i;
    int l=prob_->l;
    int w_size=get_nr_variable();
    FeatureNode **x=prob_->x;

    for(i=0;i<w_size;i++)
        Hs[i] = 0;
    for(i=0;i<l;i++)
    {
        FeatureNode * const xi=x[i];
        double xTs = sparse_operator::dot(s, xi);

        xTs = D[i]*xTs;

        sparse_operator::axpy(xTs, xi, Hs);
    }
    for(i=0;i<w_size;i++)
        Hs[i] = rho_*s[i] + Hs[i];
}

//将向量v与数据集相乘，得到向量放入Vx中。也就是权重乘以数据集
void l2r_lr_fun::Xv(double *v, double *Xv){
    int i;
    int l=prob_->l;//dataNum
    FeatureNode **x=prob_->x;//dataSize

    for(i=0;i<l;i++)
        Xv[i]=sparse_operator::dot(v, x[i]);//向量v和稀疏矩阵的第i个向量做点积，得到标量
}

//将一个元素与对应的数据集一行相乘，然后返回到XTv中，XTv累加起来
void l2r_lr_fun::XTv(double *v, double *XTv){
    int i;
    int l=prob_->l;
    int w_size=get_nr_variable();
    FeatureNode **x=prob_->x;

    for(i=0;i<w_size;i++)
        XTv[i]=0;
    for(i=0;i<l;i++)
        sparse_operator::axpy(v[i], x[i], XTv);
}

void l2r_lr_fun::get_diagH(double *M) {

}