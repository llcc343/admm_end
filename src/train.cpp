

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include "prob.h"
#include "admm.h"

int main(int argc,char **argv) {
    MPI_Init(&argc,&argv);

    //1. 参数设置
    //进程设置
    int proc_size,proc_id;
    char machine_name[1024];
    int name_len=1024;
    MPI_Comm_size(MPI_COMM_WORLD,&proc_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_id);
    MPI_Get_processor_name(machine_name,&name_len);
    std::vector<int> worker_id(proc_size-1);
    for (int i = 0; i < worker_id.size(); ++i) {
        worker_id[i]=i;
    }
    int master_id=proc_size-1;//设置master的id为最后一位
    //训练数据集设置
    char train_file[1024];
    double bias=-1.0;//bias<=0时，表示不设置bias
    //通信设置
    int partial_barrier,bounded_delay;
    //子问题求解设置
    std::string solve_sub_problem="gd";
    std::string reg="l2";
    int thread_nums;
    //ADMM超参数设置
    double rho=-1.0;
    double C=1.0;
    double abs_tol=1e-3,rel_tol=1e-3;
    int admm_iter=50;

    if(argc==5){
        thread_nums=atoi(argv[1]);
        sprintf(train_file,argv[2],proc_size,proc_id);
        partial_barrier=atoi(argv[3]);
        bounded_delay=atoi(argv[4]);
    }else{
        if(proc_id==0)
            printf("请输入：./ssp_hierarchical_parallel_subproblem_admm thread_nums train_file partial_barrier bounded_delay\n");
        return -1;
    }

    //2. 读取数据集
    Problem prob(train_file,bias);

    //3. 数据显示
    if(proc_id==master_id)
        printf("当前进程：%d 当前进程的机器:%s 测试数据集:%s\n",proc_id,machine_name,train_file);
    else
        printf("当前进程：%d 当前进程的机器:%s 训练数据集:%s\n",proc_id,machine_name,train_file);
    MPI_Barrier(MPI_COMM_WORLD);
    if(proc_id==master_id){
        printf("进程总数:%d\n",proc_size);
        printf("worker的id:");
        for (int i = 0; i < worker_id.size(); ++i) {
            printf("%d ",worker_id[i]);
        }
        printf("\n");
        printf("样本总数:%d\n",prob.l*worker_id.size());
        printf("特征个数:%d\n",prob.n);
        printf("线程数:%d\n",thread_nums);
        printf("partial_barrier:%d\n",partial_barrier);
        printf("bounded_delay:%d\n",bounded_delay);
        printf("rho=%f\n",rho);
        printf("C=%f\n",C);
        printf("abs_tol=%f\n",abs_tol);
        printf("rel_tol=%f\n",rel_tol);
        printf("admm_iter=%d\n",admm_iter);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //4. 开始训练
    Admm admm(proc_id,proc_size,master_id,worker_id,&prob,partial_barrier,bounded_delay,solve_sub_problem);
    admm.train();

    //5.结束
    MPI_Finalize();
    return 0;
}