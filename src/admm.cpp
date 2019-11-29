//
// Created by zjh on 2019/11/25.
//

#include "admm.h"


Admm::Admm(int proc_id, int proc_size, int master_id, std::vector<int> worker_id, Problem *prob,
                   int partial_barrier, int bounded_delay,
                   std::string solve_sub_problem,std::string reg,int thread_nums,
                   double rho,double C,double abs_tol,double rel_tol,int admm_iter)
        :AdmmComm(proc_id,proc_size,master_id,worker_id,partial_barrier,bounded_delay,prob->n,2*prob->n),
         solve_sub_problem_(solve_sub_problem),reg_(reg),thread_nums_(thread_nums),
         prob_(prob),rho_(rho),C_(C),abs_tol_(abs_tol),rel_tol_(rel_tol),admm_iter_(admm_iter){

    data_num_=prob_->l;
    dim_=prob_->n;

    x_=new double[dim_]();
    y_=new double[dim_]();
    z_=master_to_worker_data_;
    if(proc_id_==master_id_){
        z_pre_=new double[dim_]();
    }
    total_run=0.0;
    total_compute=0.0;
}

Admm::~Admm() {
    delete []x_;
    delete []y_;
    if(proc_id_==master_id_){
        delete []z_pre_;
    }
}

void Admm::subproblem_multicore_tron(){
    double eps = 0.001;
    double eps_cg = 0.1;
    int pos = 0;                      //label为正的个数
    int neg = 0;                      //label为负的个数
    for (int i = 0; i < prob_->l; i++) //dataNum
        if (prob_->y[i] > 0)
            pos++;
    neg = prob_->l - pos;
    double primal_solver_tol = eps * std::max(std::min(pos, neg), 1) / prob_->l;

    fun_obj_ = new l2r_lr_fun_multicore_tron(prob_,rho_); //C为特征个数个rho
    TRON *tron_obj = new TRON(fun_obj_, primal_solver_tol, eps_cg);
    tron_obj->tron(x_, y_, z_);
    delete fun_obj_;
    delete tron_obj;
}

void Admm::subproblem_gd() {
    fun_obj_=new l2r_lr_fun(prob_,rho_);
    Gd *gd_obj=new Gd(fun_obj_);
    gd_obj->train(x_,y_,z_);
    delete fun_obj_;
    delete gd_obj;
}

void Admm::x_update() {
    if(solve_sub_problem_=="multicore_tron"){
        subproblem_multicore_tron();
    }else if(solve_sub_problem_=="gd"){
        subproblem_gd();
    }
}

void Admm::y_update() {
    for (size_t i = 0; i < dim_; ++i) {
        y_[i]+=rho_*(x_[i]-z_[i]);
    }
}

// l1 or l2 or none
void Admm::z_update() {
    if(reg_=="l2"){
        for (int i = 0; i < dim_; ++i) {
            z_pre_[i]=z_[i];
            z_[i]=0.0;
            for (int j = 0; j < worker_nums_; ++j) {
                z_[i]+=master_data_[j][i]+master_data_[j][i+dim_]/rho_;
            }
            z_[i]*=rho_/(C_+worker_nums_*rho_);
        }
    }else if(reg_=="l1"){
        double k=C_/(worker_nums_*rho_);
        for(int i=0;i<dim_;++i){
            z_pre_[i]=z_[i];
            z_[i]=0.0;
            for(int j=0;j<worker_nums_;++j){
                z_[i]+=master_data_[j][i]-master_data_[j][i+dim_]/rho_;
            }
            z_[i]=z_[i]/worker_nums_;
            if(z_[i]>k){
                z_[i]=z_[i]-k;
            }else if(z_[i]<-k){
                z_[i]=z_[i]+k;
            }else{
                z_[i]=0.0;
            }
        }
    }else{
        for (int i = 0; i < dim_; ++i) {
            z_pre_[i]=z_[i];
            z_[i]=0.0;
            for (int j = 0; j < worker_nums_; ++j) {
                z_[i]+=master_data_[j][i]+master_data_[j][i+dim_]/rho_;
            }
            z_[i]=z_[i]/worker_nums_;
        }
    }
}

bool Admm::is_stop() {
    //prires
    double prires=0;
    for (int i = 0; i < worker_nums_; i++){
        for(int j=0;j<dim_;j++){
            prires+=(master_data_[i][j]-z_[j])*(master_data_[i][j]-z_[j]);
        }
    }
    prires=sqrt(prires);
    //dualres
    double zdiff=0.0;
    double z_squrednorm = 0.0;
    for(int i=0;i<dim_;++i){
        zdiff+=(z_[i]-z_pre_[i])*(z_[i]-z_pre_[i]);
        z_squrednorm += z_[i] * z_[i];
    }
    double dualres=sqrt(worker_nums_)*rho_*sqrt(zdiff);
    //eps_pri
    double nxstack=0;
    for (size_t i = 0; i < worker_nums_; i++){
        for (size_t j = 0; j < dim_; j++){
            nxstack+=master_data_[i][j]*master_data_[i][j];
        }
    }
    nxstack=sqrt(nxstack);
    double z_norm=sqrt(worker_nums_*z_squrednorm);
    double eps_pri=sqrt(worker_nums_*dim_)*abs_tol_+rel_tol_*fmax(nxstack,z_norm);
    //eps_dual
    double nystack=0;
    for (size_t i = 0; i < worker_nums_; i++){
        for (size_t j = 0; j < dim_; j++){
            nystack+=master_data_[i][j+dim_]*master_data_[i][j+dim_];
        }
    }
    nystack=rho_*sqrt(nystack);
    double eps_dual=sqrt(worker_nums_*dim_)*abs_tol_+rel_tol_*nystack;

    printf("%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", prires, eps_pri, dualres, eps_dual,current_cost_function(),current_accuracy());

    if ((prires<=eps_pri) && (dualres<=eps_dual)){
        return true;
    }
    return false;
}

double Admm::current_accuracy(){
    std::vector<double> hypothesis(data_num_,0);//得到sigmoid函数的值
    //1计算每个数据中的sigmoid函数的值
    for(int i=0;i<data_num_;i++){
        int j=0;
        while(true){
            if(prob_->x[i][j].index==-1)
                break;
            hypothesis[i]+=-prob_->x[i][j].value*z_[prob_->x[i][j].index-1];
            j++;
        }
        hypothesis[i]=1/(1+exp(hypothesis[i]));
    }
    //2计算accuracy
    double acc=0;
    for(int i=0;i<data_num_;i++){
        if(hypothesis[i]>0.5&&prob_->y[i]>0.5)
            acc++;
        else if(hypothesis[i]<0.5&&prob_->y[i]<0.5)
            acc++;
    }
    acc=acc/double(data_num_);
    return acc;
}

double Admm::current_cost_function(){
    std::vector<double> hypothesis(data_num_, 0); //存储sigmoid函数的值，每个值表示每条数据对应的sigmoid值
    //1计算每个数据中的sigmoid函数的值，得到的值保存在hypothesis中
    for (int i = 0; i < data_num_; i++){
        int j = 0;
        while (true){
            if (prob_->x[i][j].index == -1)
                break;
            hypothesis[i] += -prob_->x[i][j].value * z_[prob_->x[i][j].index - 1];
            j++;
        }
        hypothesis[i] = 1 / (1 + exp(hypothesis[i]));
    }
    //2计算costFunction
    double costFunction = 0.0;
    for (int i = 0; i < data_num_; i++){
        if (prob_->y[i] == 1){
            costFunction += log(hypothesis[i]);
        }
        else{
            costFunction += log(1 - hypothesis[i]);
        }
    }
    costFunction = -costFunction / data_num_;
    return costFunction;
}

bool Admm::MasterUpdate(int k) {
    z_update();
    printf("%d",k);
    if(is_stop()||k>admm_iter_){
        return true;
    }else{
        return false;
    }
}

void Admm::WorkerUpdate(int k){
    double start,end;
    start=omp_get_wtime();
    if(k!=0){
        y_update();
    }
    x_update();
    for (size_t i = 0; i < 2*dim_; ++i) {
        if(i<dim_){
            worker_to_master_data_[i]=x_[i];
        }else{
            worker_to_master_data_[i]=y_[i-dim_];
        }
    }
    end=omp_get_wtime();
    total_compute+=(end-start);
}

void Admm::train() {
    omp_set_num_threads(thread_nums_);
    double start,end;
    if(proc_id_==master_id_){
        printf("%3s %10s %10s %10s %10s %10s %10s\n", "#", "r norm", "esp_pri", "s norm", "esp_dual","objective","accuracy");
        RunMaster();
    }
    else{
        start=omp_get_wtime();
        RunWorker();
        end=omp_get_wtime();
        total_run=(end-start);
    }

    double all_total_run,all_total_comm,all_total_compute;

    MPI_Allreduce(&total_run,&all_total_run,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(&total_compute,&all_total_compute,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(&comm_time,&all_total_comm,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    all_total_compute/=worker_nums_;
    all_total_run/=worker_nums_;
    all_total_comm/=worker_nums_;
    if(proc_id_==master_id_){
        printf("通信时间为:%f\n",all_total_comm);
        printf("计算时间为:%f\n",all_total_compute);
        printf("总运行时间为:%f\n",all_total_run);
    }
}
