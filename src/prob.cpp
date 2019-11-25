//
// Created by zjh on 2019/11/25.
//

#include "prob.h"


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))//分配n个type类型的存储单元，并返回首地址

Problem::Problem(const char *filename,double bias):filename_(filename),bias_(bias){
    read_problem();
}

void Problem::read_problem(){
    //1.打开文件
    int max_index, inst_max_index, i;
    size_t elements, j;//size_t是无符号整形，typedef unsigned int size_t
    FILE *fp = fopen(filename_,"r");
    char *endptr;
    char *idx, *val, *label;

    if(fp == NULL){
        fprintf(stderr,"can't open input file %s\n",filename_);
        exit(1);
    }

    l = 0;
    elements = 0;
    max_line_len = 1024;
    line = Malloc(char,max_line_len);//为line分配1024个char类型的存储单元，返回这个存储单元的首地址
    //2.读取每一行的数据,计算行数，数据个数
    while(readline(fp)!=NULL)//读取数据集中的一行数据，直到没有数据
    {
        // label，切割字符串，将str切分成一个个子串，根据字符'\t'，将字符'\t'替换成'\0'
        char *p = strtok(line," \t");

        // features，这里的elements统计每一行有多少个非稀疏的数据
        while(1){
            //这里的strtok每执行一次，就得到一个子字符串
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                break;
            elements++;
        }
        elements++; // for bias term
        l++;//统计一个文件中有多少行
    }
    rewind(fp);//使文件fp的位置指针指向文件开始

    y = Malloc(double,l);
    x = Malloc(struct FeatureNode *,l);
    x_space = Malloc(struct FeatureNode,elements+l);
    // printf("elem+l:%d\n",elements+l);
    max_index = 0;
    j=0;
    //读取每行数据，将label放入y，将数据放入x_space
    for(i=0;i<l;i++)//循环l次，而l=数据集中数据的条数
    {
        inst_max_index = 0; // strtol gives 0 if wrong format
        readline(fp);//读取数据集中的一行数据，并将这行数据放入到line中
        x[i] = &x_space[j];//这里的x[i]为指针，x为二级指针
        label = strtok(line," \t\n");
        if(label == NULL) // empty line
            exit_input_error(i+1);

        y[i] = strtod(label,&endptr);//功能是将字符串转换成浮点数,标签是否需要将-1转换成0
        // y[i]=(y[i]==-1) ? 0.0 : 1.0;?
        if(endptr == label || *endptr != '\0')
            exit_input_error(i+1);

        while(1){
            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            if(val == NULL)
                break;

            errno = 0;
            x_space[j].index = (int) strtol(idx,&endptr,10);
            if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
                exit_input_error(i+1);
            else
                inst_max_index = x_space[j].index;

            errno = 0;
            x_space[j].value = strtod(val,&endptr);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(i+1);

            ++j;
        }

        if(inst_max_index > max_index)
            max_index = inst_max_index;

        if(bias_ >= 0)
            x_space[j++].value = bias_;

        x_space[j++].index = -1;
    }

    MPI_Allreduce(&max_index,&n,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);

    if(bias_ >= 0){
        n=n+1;
        for(i=1;i<l;i++)
            (x[i]-2)->index = n;
        x_space[j-2].index = n;
    }
    fclose(fp);
}
//读取input中的行，将其返回到*line中
char * Problem::readline(FILE *input){
    int len;

    //从文件结构提指针input中读取数据，每次读取一行．
    //读取的数据保存在line里，如果文件中的该行，不足bufsize-1个字符，则读完该行就结束。
    //如若该行（包括最后一个换行符）的字符数超过bufsize-1，
    //则fgets只返回一个不完整的行，但是，缓冲区总是以NULL字符结尾，对fgets的下一次调用会继续读该行。
    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    //查找字符'\n'在line中出现的位置
    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    //这里的line是每一行的数据
    return line;
}

void Problem::exit_input_error(int line_num){
    fprintf(stderr,"Wrong input format at line %d\n", line_num);
    exit(1);
}