//
// Created by zjh on 2019/11/25.
//

#ifndef ADMM_END_ADMM_COMM_H
#define ADMM_END_ADMM_COMM_H

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <vector>

class AdmmComm {
public:
    AdmmComm(int proc_id, int proc_size, int master_id, std::vector<int> worker_id,
        int partial_barrier, int bounded_delay,
        int master_to_worker_data_size, int worker_to_master_data_size);
    virtual ~AdmmComm();

    void RunMaster();

    void RunWorker();

    virtual bool MasterUpdate(int k) = 0;
    virtual void WorkerUpdate(int k) = 0;

    virtual void train();

protected:
    int proc_id_;
    int proc_size_;
    int master_id_;
    std::vector<int> worker_id_;
    size_t worker_nums_;

    int partial_barrier_, bounded_delay_;

    //size is (worker_nums,worker_to_master_data_size),save every worker's update,worker 0 in master_data[0][]
    double **master_data_;

    double *master_to_worker_data_;
    int master_to_worker_data_size_;

    double *worker_to_master_data_;
    int worker_to_master_data_size_;

    double comm_time;

private:
};


#endif //ADMM_END_ADMM_COMM_H
