//
// Created by zjh on 2019/11/25.
//

#ifndef ADMM_END_PROB_H
#define ADMM_END_PROB_H

#include <mpi.h>
#include <cstring>
#include <iostream>

struct FeatureNode{
    int index;
    double value;
};

class Problem {
public:
    Problem(const char *filename,double bias);
    void read_problem();
    char* readline(FILE *input);
    void exit_input_error(int line_num);

    int l, n;//数据量,特征的个数
    double *y;//标签
    struct FeatureNode **x;//数据集
    struct FeatureNode *x_space;
    double bias_;            /* < 0 if no bias term */
    char *line;
    int max_line_len;
    const char *filename_;
};


#endif //ADMM_END_PROB_H
