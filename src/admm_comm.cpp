//
// Created by zjh on 2019/11/25.
//

#include "admm_comm.h"

AdmmComm::AdmmComm(int proc_id, int proc_size, int master_id, std::vector<int> worker_id, int partial_barrier,
         int bounded_delay, int master_to_worker_data_size, int worker_to_master_data_size)
        : proc_id_(proc_id), proc_size_(proc_size), master_id_(master_id), worker_id_(worker_id),
          partial_barrier_(partial_barrier), bounded_delay_(bounded_delay),
          master_to_worker_data_size_(master_to_worker_data_size), worker_to_master_data_size_(worker_to_master_data_size){

    worker_nums_ = worker_id.size();

    if (proc_id_ == master_id_){
        master_data_ = new double *[worker_nums_];
        for (int i = 0; i < worker_nums_; ++i){
            master_data_[i] = new double[worker_to_master_data_size_]();
        }
    }
    master_to_worker_data_ = new double[master_to_worker_data_size_]();
    worker_to_master_data_ = new double[worker_to_master_data_size_]();

    comm_time = 0.0;
}

AdmmComm::~AdmmComm(){
    delete[] master_to_worker_data_;
    delete[] worker_to_master_data_;
    if (proc_id_ == master_id_)
    {
        for (int i = 0; i < worker_nums_; ++i)
        {
            delete[] master_data_[i];
        }
        delete[] master_data_;
    }
}

void AdmmComm::RunMaster(){
    int k = 0;
    bool is_stop = false;
    //1. 初始化循环非阻塞接收
    MPI_Request *recv_request = new MPI_Request[worker_nums_];
    for (int i = 0; i < worker_nums_; ++i){
        MPI_Recv_init(&master_data_[i][0], worker_to_master_data_size_, MPI_DOUBLE, worker_id_[i], 0, MPI_COMM_WORLD, &recv_request[i]);
    }
    std::vector<bool> is_recv(worker_nums_, false);
    int recv_nums = 0;
    std::vector<int> t_worker(worker_nums_, 1);//每个worker进程上的tao值，初始为1，当某个tao值大于等于设定的有界延迟时，本次需要接受这个进程的更新
    int indx = 0;
    MPI_Status recv_status;
    std::vector<int> recv_worker_id;
    //2. 循环
    while (true){
        //3. 判断每个非阻塞接收是否启动，未启动的启动
        for (int i = 0; i < worker_nums_; ++i){
            if (!is_recv[i]){
                MPI_Start(&recv_request[i]);
                is_recv[i] = true;
            }
        }
        //4. ssp条件的接收
        while (recv_nums < partial_barrier_ || *std::max_element(t_worker.cbegin(), t_worker.cend()) >= bounded_delay_){
            MPI_Waitany(worker_nums_, recv_request, &indx, &recv_status);//indx返回recv_request中的第indx个，也就是索引
            recv_nums++;
            is_recv[indx] = false;
            t_worker[indx] = 0;
            // recv_worker_id.push_back(worker_id_[indx]);
            recv_worker_id.push_back(recv_status.MPI_SOURCE);
        }

        for (int i = 0; i < worker_nums_; ++i){
            ++t_worker[i];
        }

        recv_nums = 0;
        //5. master上的数据更新
        is_stop = MasterUpdate(k);

        //6. 判断是否停止
        if (is_stop){
            for (int i = 0; i < worker_nums_; ++i){
                MPI_Request_free(&recv_request[i]);
            }
            for (int i = 0; i < master_to_worker_data_size_; ++i){
                master_to_worker_data_[i] = 0;
            }
            for (int i = 0; i < worker_nums_; ++i){
                MPI_Send(master_to_worker_data_, master_to_worker_data_size_, MPI_DOUBLE, worker_id_[i], 1, MPI_COMM_WORLD);
            }
            break;
        }

        //7. 发送消息给部分worker
        for (int i = 0; i < recv_worker_id.size(); ++i){
            MPI_Send(master_to_worker_data_, master_to_worker_data_size_, MPI_DOUBLE, recv_worker_id[i], 1, MPI_COMM_WORLD);
        }
        recv_worker_id.clear();
        ++k;
    }
}

void AdmmComm::RunWorker(){
    int k = 0;
    double start, end;
    while (true){
        //0. 更新本地信息
        WorkerUpdate(k);
        start = omp_get_wtime();
        //1. worker 发送更新信息到master
        MPI_Send(worker_to_master_data_, worker_to_master_data_size_, MPI_DOUBLE, master_id_, 0, MPI_COMM_WORLD);
        //2. worker等待从master接收更新信息
        MPI_Status recv_status;
        MPI_Recv(master_to_worker_data_, master_to_worker_data_size_, MPI_DOUBLE, master_id_, 1, MPI_COMM_WORLD, &recv_status);
        end = omp_get_wtime();
        comm_time += (end - start);
        double sum = 0;
        for (int i = 0; i < master_to_worker_data_size_; ++i){
            sum += master_to_worker_data_[i];
        }
        ++k;
        if (sum == 0){
            break;
        }
    }
}

void AdmmComm::train(){
    if (proc_id_ == master_id_)
        RunMaster();
    else
        RunWorker();
}
