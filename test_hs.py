from util_scheduling import *
from collections import defaultdict,deque
import statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def find_dict_parents(dag):
    """
    dag: Dict
    """
    dict_parent = defaultdict(set)
    for task,list_children in dag.items():
        for child in list_children:
            dict_parent[child].add(task)
    return dict_parent

def cal_makespan_mean(small_DAG,dict_process,dict_communication,assignment):
    """
    small_DAG : Dict (Assume starting task is min index of tasks)
        dictionary of dag of task graph (indices start from 1)
    p : Dict
        dictionary of required computations for each task
    e: Dict (indices start from 1)
        dictionary of exec. speed of machines
    B: Dict 
        bandwidth between machines 
    d: Dict
        dictionary of generated data after exec. each task 
    assignment: Dict (key=task:value=index machine start from 0) 
    """
    dict_parent = find_dict_parents(small_DAG)
    
    dict_machines_busy_time, dict_start, dict_finish, dict_rec_corresp_data = {m:0.0 for m in range(len(dict_communication[min(dict_communication.keys())]))}, {k:0.0 for k in assignment.keys()}, {k:0.0 for k in assignment.keys()}, {k:0.0 for k in assignment.keys()}
    
    start_task,end_task = min(assignment.keys()), max(assignment.keys())
    ## key idea: the task with minimum REMAINING indegree is popped from the heap 
    q = deque([start_task]) # (indegree_task,task)
    while q:
        task = q.popleft() # pop means it is gonna be executed on a machine
        dict_finish[task] = dict_start[task]+dict_process[task][assignment[task]]#(p[task]/e[assignment[task]])
        dict_machines_busy_time[assignment[task]] = dict_finish[task]
        for child in small_DAG[task]:
            #if dict_parent[child]:
            dict_rec_corresp_data[child] = max(dict_rec_corresp_data[child], dict_finish[task]+ dict_communication[task][assignment[task]][assignment[child]] ) # if assignment[task]!=assignment[child] else 0.0
            #print(f"- for loop child {child} (on {assignment[child]}) parent {task} (on {assignment[task]}) {dict_rec_corresp_data[child]} bandw. {Bandwith[assignment[task]][assignment[child]]} ")
            dict_parent[child].remove(task)
            #print(f"after-- dict_parent[child] {dict_parent[child]}")
            if not dict_parent[child]:
                #print(f"---> no upward conn. for {child} with rec time {dict_rec_corresp_data[child]}")
                q.append(child)

                dict_start[child] = max(dict_rec_corresp_data[child],dict_machines_busy_time[assignment[child]])
                #print(f"---> start time {dict_start[child]}")

        #print(f"f task {task}: s {dict_start[task]} f {dict_finish[task]}")
    #print(f"INSIDE cal {dict_finish}")
    return dict_finish[end_task]

def cal_makespan(small_DAG,p,d,e,Bandwith,assignment):
    """
    small_DAG : Dict (Assume starting task is min index of tasks)
        dictionary of dag of task graph (indices start from 1)
    p : Dict
        dictionary of required computations for each task
    e: Dict (indices start from 1)
        dictionary of exec. speed of machines
    B: Dict 
        bandwidth between machines 
    d: Dict
        dictionary of generated data after exec. each task 
    assignment: Dict (key=task:value=index machine start from 0) 
    """
    dict_parent = find_dict_parents(small_DAG)
    
    dict_machines_busy_time, dict_start, dict_finish, dict_rec_corresp_data = {m:0.0 for m in e.keys()}, {k:0.0 for k in p.keys()}, {k:0.0 for k in p.keys()}, {k:0.0 for k in p.keys()}
    
    start_task,end_task = min(p.keys()), max(p.keys())
    ## key idea: the task with minimum REMAINING indegree is popped from the heap 
    q = deque([start_task]) # (indegree_task,task)
    while q:
        task = q.popleft() # pop means it is gonna be executed on a machine
        dict_finish[task] = dict_start[task]+(p[task]/e[assignment[task]])
        dict_machines_busy_time[assignment[task]] = dict_finish[task]
        for child in small_DAG[task]:
            #if dict_parent[child]:
            dict_rec_corresp_data[child] = max(dict_rec_corresp_data[child], dict_finish[task]+ (d[task]/Bandwith[assignment[task]][assignment[child]]) ) # if assignment[task]!=assignment[child] else 0.0
            #print(f"- for loop child {child} (on {assignment[child]}) parent {task} (on {assignment[task]}) {dict_rec_corresp_data[child]} bandw. {Bandwith[assignment[task]][assignment[child]]} ")
            dict_parent[child].remove(task)
            #print(f"after-- dict_parent[child] {dict_parent[child]}")
            if not dict_parent[child]:
                #print(f"---> no upward conn. for {child} with rec time {dict_rec_corresp_data[child]}")
                q.append(child)

                dict_start[child] = max(dict_rec_corresp_data[child],dict_machines_busy_time[assignment[child]])
                #print(f"---> start time {dict_start[child]}")

        #print(f"f task {task}: s {dict_start[task]} f {dict_finish[task]}")
    #print(f"INSIDE cal {dict_finish}")
    return dict_finish[end_task]


def heft(small_DAG,p,d,e,Bandwith):
    """
    small_DAG : Dict (Assume starting task is min index of tasks)
        dictionary of dag of task graph (indices start from 1)
    p : Dict
        dictionary of required computations for each task
    e: Dict (indices start from 1)
        dictionary of exec. speed of machines
    B: Dict 
        bandwidth between machines 
    d: Dict
        dictionary of generated data after exec. each task 
    """
    #print("=======================   new running of HEFT ==========")
    def commcost(ni, nj, A, B):
        # if(A==B):
        #     return 0
        # else:
        #     return d[ni]/Bandwith[ord(A)-97][ord(B)-97] #average_com_across_nodes
        #print(f"ni {ni} nj {nj} A {A} B {B}")
        return d[ni]/Bandwith[ord(A)-97][ord(B)-97]
    def compcost(job,agent):
        x = p[job]/ e[ord(agent)-97]   
        return x
 
    string_naming_machines_with_alphabets = ''
    for i in range(len(e.keys())):
        string_naming_machines_with_alphabets += chr(97+i)

    orders, jobson = schedule(small_DAG, string_naming_machines_with_alphabets, compcost, commcost)
    return orders, jobson

def heft_mean_based(small_DAG,dict_process,dict_communication):
    """
    small_DAG : Dict (Assume starting task is min index of tasks)
        dictionary of dag of task graph (indices start from 1)
    p : Dict
        dictionary of required computations for each task
    e: Dict (indices start from 1)
        dictionary of exec. speed of machines
    B: Dict 
        bandwidth between machines 
    d: Dict
        dictionary of generated data after exec. each task 
    """
    def commcost(ni, nj, A, B):
        # if(A==B):
        #     return 0
        # else:
        #     return dict_communication[ni][ord(A)-97][ord(B)-97] #d[ni]/Bandwith[ord(A)-97+1][ord(B)-97+1] #average_com_across_nodes
        return dict_communication[ni][ord(A)-97][ord(B)-97]
    def compcost(job,agent):
        x = dict_process[job][ord(agent)-97]#p[job]/ e[ord(agent)-97+1]
        return x

    start_task = min(dict_process.keys())
    num_all_machines = len(dict_process[start_task].values())
    string_naming_machines_with_alphabets = ''
    for i in range(num_all_machines):
        string_naming_machines_with_alphabets += chr(97+i)
    orders, jobson = schedule(small_DAG, string_naming_machines_with_alphabets, compcost, commcost)
    return orders,jobson

def find_stats_div_p_over_e(list_dict_p,list_dict_e):
    list_tasks,list_machines = list(list_dict_p[0].keys()),list(list_dict_e[0].keys())
    ## -- initialize dict_proc 
    dict_proc = {t:dict() for t in list_tasks}
    for t in list_tasks:
        for m in list_machines:
            dict_proc[t][m] = statistics.mean([list_dict_p[i][t]/list_dict_e[i][m] for i in range(len(list_dict_p))])      
    ## -- est. variance of data
    dict_proc_var = {t:dict() for t in list_tasks}
    for t in list_tasks:
        for m in list_machines:
            dict_proc_var[t][m] = statistics.variance([list_dict_p[i][t]/list_dict_e[i][m] for i in range(len(list_dict_p))])
    #print(f"dict_proc {dict_proc}")
    
    ## --- for plotting the histogram
    if len(list_tasks)<10:
        fig, axs = plt.subplots(nrows=len(list_tasks), ncols=len(list_machines), figsize=(12, 8)) #
        ## because tasks starts from index 1, subtract by 1
        for t in list_tasks:
            for m in list_machines:
                division_p_over_e = [list_dict_p[i][t]/list_dict_e[i][m] for i in range(len(list_dict_p))]
                mean,std = np.round(dict_proc[t][m],2), np.round(np.sqrt(dict_proc_var[t][m]),2)
                axs[t-1, m].hist(division_p_over_e, bins=20,alpha=0.5)
                axs[t-1, m].axvline(mean, color='red', linestyle='dashed', linewidth=2)
                axs[t-1, m].axvline(mean - std, color='orange', linestyle='dashed', linewidth=2)
                axs[t-1, m].axvline(mean + std, color='orange', linestyle='dashed', linewidth=2)
                axs[t-1, m].set_title(f"task {t} machine {m} (mu={mean},std={std})",fontdict={'fontsize': 8})
        plt.tight_layout()
        #fig.subplots_adjust(left=0.1, bottom=0.1, right=1.2, top=1.2, wspace=0.05, hspace=0.05)
        fig.savefig('full_figure.png')
    
    return dict_proc, dict_proc_var

def find_stats_div_d_over_B(list_dict_d,list_list_B):
    list_tasks,list_machines = list(list_dict_d[0].keys()),list(range(len(list_list_B[0])))
    ## -- initialize dict_comm 
    dict_comm = {t:[[0.0 for _ in list_machines] for _ in list_machines] for t in list_tasks}
    dict_comm_var = {t:[[0.0 for _ in list_machines] for _ in list_machines] for t in list_tasks}
    
    for t in list_tasks:
        for m in list_machines:
            for n in list_machines:
                dict_comm[t][m][n] = statistics.mean([(list_dict_d[i][t]/list_list_B[i][m][n]) for i in range(len(list_dict_d))])
    ## -- for est. variance of data
    for t in list_tasks:
        for m in list_machines:
            for n in list_machines:
                dict_comm_var[t][m][n] = statistics.variance([(list_dict_d[i][t]/list_list_B[i][m][n]) for i in range(len(list_dict_d))])
    #print(f"dict_comm {dict_comm}")
    return dict_comm, dict_comm_var

def find_stat_dict(list_dict):
    dict_mean = {k:statistics.mean(
        [x[k] for x in list_dict]
    ) for k in list_dict[0].keys()}
    dict_var = {k:statistics.variance(
        [x[k] for x in list_dict]
    ) for k in list_dict[0].keys()}
    return dict_mean,dict_var

def find_stat_2Dlist(x):
    res_mean = [[statistics.mean(x[it][i][j] for it in range(len(x))) for j in range(len(x[0][0]))] for i in range(len(x[0]))]
    res_var = [[statistics.variance(x[it][i][j] for it in range(len(x))) for j in range(len(x[0][0]))] for i in range(len(x[0]))]
    return res_mean, res_var
if __name__=="__main__":
    # machine indices start from 0
    num_machines = 4
    iterations = 100
    num_diff_stats = 1
    
    
    task_graph = multi_chain(2,2)#obj_and_pose_recognition_task_graph()#multi_chain(10,5)#multi_chain(20,10)#
    
    end_task = max(task_graph.keys())

    ave_all_makespan = 0.0
    ave_all_makespan_based_mean_assign = 0.0
    ave_all_makespan_based_mean_var_assign = 0.0
    ave_all_makespan_sheft = 0.0

    for idx_setting_stats in range(num_diff_stats):
        stat_d = {k:[np.random.uniform(low=10.0, high=100.0),10*np.random.rand()] for k in task_graph.keys()}
        stat_p = {k:[np.random.uniform(low=10.0, high=100.0),10*np.random.rand()] for k in task_graph.keys()}
        stat_e = {k:[np.random.uniform(low=10.0, high=100.0),10*np.random.rand()] for k in range(num_machines)}
        stat_B_list = [[[np.random.uniform(low=10.0, high=100.0),10*np.random.rand()] for _ in range(num_machines)] for _ in range(num_machines)]#
        for i in range(num_machines):
            stat_B_list[i][i] = [10000.0,10*np.random.rand()] 
        #print(f"all stats -- \n stat_d {stat_d} \n stat_p {stat_p} \n stat_e {stat_e} \n stat_B_list {stat_B_list}")
        ## ---- make realizatiion of parameters based on above stats 
        list_dict_p = [{k:abs(np.random.normal(stat_p[k][0],stat_p[k][1])) for k in task_graph.keys()} for _ in range(iterations)]
        list_dict_e = [{m:abs(np.random.normal(stat_e[m][0],stat_e[m][1])) for m in range(num_machines)} for _ in range(iterations)]
        dict_process, dict_process_var = find_stats_div_p_over_e(list_dict_p,list_dict_e)

        list_dict_d = [{k:abs(np.random.normal(stat_d[k][0],stat_d[k][1])) for k in task_graph.keys()} for _ in range(iterations)]
        list_list_B = [[[abs(np.random.normal(stat_B_list[i][j][0],stat_B_list[i][j][1])) for i in range(num_machines)] for j in range(num_machines)] for _ in range(iterations) ]
        dict_communication, dict_communication_var = find_stats_div_d_over_B(list_dict_d,list_list_B)
        
        ## --  calculate mean and variance from data
        dict_p_mean,dict_p_var = find_stat_dict(list_dict_p)
        dict_d_mean,dict_d_var = find_stat_dict(list_dict_d)
        dict_e_mean,dict_e_var = find_stat_dict(list_dict_e)
        B_mean, B_var =  find_stat_2Dlist(list_list_B)

        ## ----- makespan of heft on mean of division 
        order_mean, jobson_mean = heft_mean_based(task_graph,dict_process,dict_communication)
        assignment_mean = {k:ord(v)-97 for k,v in jobson_mean.items()}
        makespan_mean = cal_makespan_mean(task_graph,dict_process,dict_communication,assignment_mean)
        
        # ----- makespan of heft on mean PLUS sqrt(variance)
        dict_process_mean_var = dict_process
        for k in dict_process_mean_var.keys():
            for m in dict_process_mean_var[k].keys():
                dict_process_mean_var[k][m] += np.sqrt(dict_process_var[k][m])
        
        dict_communication_mean_var = dict_communication
        for k in dict_communication_mean_var.keys():
            for m in range(num_machines):
                for n in range(num_machines):
                    dict_communication_mean_var[k][m][n] += np.sqrt(dict_communication_var[k][m][n])
        
        order_mean_var, jobson_mean_var = heft_mean_based(task_graph,dict_process_mean_var,dict_communication_mean_var)
        assignment_mean_var = {k:ord(v)-97 for k,v in jobson_mean_var.items()}
        
        makespan_mean_var = cal_makespan_mean(task_graph,dict_process_mean_var,dict_communication_mean_var,assignment_mean_var)

        for k,v in order_mean.items():
            for eve in v:
                if eve.job==end_task:
                    makespan_mean = eve.end
                    #print(f"this it finishing time {eve.end}")
        
        ## -- SHEFT assignment
        p_sheft = {k:dict_p_mean[k]+np.sqrt(dict_p_var[k]) for k in task_graph.keys()}
        d_sheft = {k:dict_d_mean[k]+np.sqrt(dict_d_var[k]) for k in task_graph.keys()}
        e_sheft = dict_e_mean
        B_sheft = B_mean
        order_mean_SHEFT, jobson_mean_SHEFT = order_it_sheft, jobson_it_sheft = heft(task_graph,p_sheft,d_sheft,e_sheft,B_sheft)
        assignment_mean_SHEFT = {k:ord(v)-97 for k,v in jobson_mean_SHEFT.items()}
        
        makespan_mean_SHEFT = cal_makespan(task_graph,p_sheft,d_sheft,e_sheft,B_sheft,assignment_mean_SHEFT)
        
        ## ---- realization 
        all_makespan = [0.0 for _ in range(iterations)]
        all_makespan_based_mean_assign = [0.0 for _ in range(iterations)]
        all_makespan_based_mean_var_assign = [0.0 for _ in range(iterations)]
        all_makespan_sheft = [0.0 for _ in range(iterations)]
        for it in range(iterations):
            if it%1000 == 0:
                print(f"idx setting stats {idx_setting_stats} iteration {it} ...")
            d = list_dict_d[it]#{k:abs(np.random.normal(stat_d[k][0],stat_d[k][1])) for k in task_graph.keys()}#{1:4.0,2:2.0,3:3.0,4:2.0,5:1.0,6:7.5}#
            p = list_dict_p[it]#{k:abs(np.random.normal(stat_p[k][0],stat_p[k][1])) for k in task_graph.keys()}#{1:1.0,2:2.0,3:1.0,4:2.0,5:5.0,6:4.0}#
            e = list_dict_e[it]#{m:abs(np.random.normal(stat_e[m][0],stat_e[m][1])) for m in range(num_machines)}#{0:1.0,1:1.0}#
            B_list = list_list_B[it]#[[ abs(np.random.normal(stat_B_list[i][j][0],stat_B_list[i][j][0]) ) for i in range(num_machines)] for j in range(num_machines)]#[[10.0,1.0],[3.0,20.0]]#
            
            ## -- actual HEFT
            order_it, jobson_it = heft(task_graph,p,d,e,B_list)
            assignment = {k:ord(v)-97 for k,v in jobson_it.items()}
            calculated_makespan = cal_makespan(task_graph,p,d,e,B_list,assignment)
            all_makespan[it] = calculated_makespan

            ## -- mean-based
            calculated_makespan_based_mean_assign = cal_makespan(task_graph,p,d,e,B_list,assignment_mean)
            all_makespan_based_mean_assign[it] = calculated_makespan_based_mean_assign

            ## -- mean_var-based
            calculated_makespan_based_mean_var_assign = cal_makespan(task_graph,p,d,e,B_list,assignment_mean_var)
            all_makespan_based_mean_var_assign[it] = calculated_makespan_based_mean_var_assign

            ## -- SHEFT ---
            calculated_makespan_sheft = cal_makespan(task_graph,p,d,e,B_list,assignment_mean_SHEFT)
            all_makespan_sheft[it] = calculated_makespan_sheft

            
        ave_all_makespan += sum(all_makespan)/ (1.0*iterations)
        ave_all_makespan_based_mean_assign += sum(all_makespan_based_mean_assign)/(1.0*iterations)
        ave_all_makespan_based_mean_var_assign += sum(all_makespan_based_mean_var_assign)/(1.0*iterations)
        ave_all_makespan_sheft += sum(all_makespan_sheft)/(1.0*iterations)
        

    ave_all_makespan /= (1.0*num_diff_stats)
    ave_all_makespan_based_mean_assign /= (1.0*num_diff_stats)
    ave_all_makespan_based_mean_var_assign /= (1.0*num_diff_stats)
    ave_all_makespan_sheft /= (1.0*num_diff_stats)
    print(f"expected makespan {ave_all_makespan} and all_makespan_based_mean_assign {ave_all_makespan_based_mean_assign} all_makespan_sheft {ave_all_makespan_sheft} all_makespan_based_mean_var_assign {ave_all_makespan_based_mean_var_assign}")
    
    plt.bar(range(iterations), all_makespan_sheft, color='b', align='center',label='SHEFT')
    plt.bar(range(iterations), all_makespan_based_mean_assign, color='g', align='center',label='Proposed 1')
    plt.bar(range(iterations), all_makespan_based_mean_var_assign, color='r', align='center',label='Proposed 2')
    plt.title('Multiple Curves in a Single Plot')
    plt.xlabel('Iteration')
    plt.ylabel('Makespan')

    # add legends
    plt.legend()

    # save the plot as a PNG file
    plt.savefig('plot.png')
    #print(f"makespan_mean: {makespan_mean} makespan_mean_var {makespan_mean_var} ")
    