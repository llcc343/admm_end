//
// Created by zjh on 2019/11/25.
//

#include "utils/sparse_operator.h"

Reduce_Vectors::Reduce_Vectors(int size){
    nr_thread = omp_get_max_threads();
    this->size = size;
    tmp_array = new double*[nr_thread];
    for(int i = 0; i < nr_thread; i++)
        tmp_array[i] = new double[size];
}

Reduce_Vectors::~Reduce_Vectors(void){
    for(int i = 0; i < nr_thread; i++)
        delete[] tmp_array[i];
    delete[] tmp_array;
}

void Reduce_Vectors::init(void){
#pragma omp parallel for schedule(static)
    for(int i = 0; i < size; i++)
        for(int j = 0; j < nr_thread; j++)
            tmp_array[j][i] = 0.0;
}

void Reduce_Vectors::sum_scale_x(double scalar, FeatureNode *x){
    int thread_id = omp_get_thread_num();

    sparse_operator::axpy(scalar, x, tmp_array[thread_id]);
}

void Reduce_Vectors::reduce_sum(double* v){
#pragma omp parallel for schedule(static)
    for(int i = 0; i < size; i++)
    {
        v[i] = 0;
        for(int j = 0; j < nr_thread; j++)
            v[i] += tmp_array[j][i];
    }
}