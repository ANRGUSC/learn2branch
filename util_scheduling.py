import pickle
import os
import gzip
from cire_heft import (wbar, cbar, ranku, schedule, Event, start_time,makespan)
import networkx as nx

def multi_chain(num_tasks_each_col,num_col):
    dic_task_graph = dict()
    dic_task_graph[1] = [x for x in range(2,num_col+2)]
    for x in range(2,num_col*(num_tasks_each_col-1)+2):
        dic_task_graph[x] = [x+num_col]
    for x in range(num_col*(num_tasks_each_col-1)+2,num_col*num_tasks_each_col+2):
        dic_task_graph[x] = [num_col*num_tasks_each_col+2]
    dic_task_graph[num_col*num_tasks_each_col+2]=[]
    #print("Task Graph multi_chain\n",dic_task_graph)
    return dic_task_graph

def face_recognition_task_graph():
    name_dict = {"Source":1,"Copy":2,"Tiler":3,"Detect1":4,"Detect2":5,"Detect3":6,"Feature merger":7,"Graph Spiltter":8,"Classify1":9,"Classify2":10
    ,"Reco. Merge":11,"Display":12}
    dic_task_graph = dict()
    dic_task_graph[name_dict["Source"]] = [name_dict["Copy"]]
    dic_task_graph[name_dict["Copy"]] = [ name_dict["Tiler"],name_dict["Feature merger"],name_dict["Display"]]
    dic_task_graph[name_dict["Tiler"]] = [ name_dict["Detect1"],name_dict["Detect2"],name_dict["Detect3"] ]
    dic_task_graph[name_dict["Detect1"]] = [name_dict["Feature merger"]]
    dic_task_graph[name_dict["Detect2"]] = [name_dict["Feature merger"]]
    dic_task_graph[name_dict["Detect3"]] = [name_dict["Feature merger"]]
    dic_task_graph[name_dict["Feature merger"]] = [name_dict["Graph Spiltter"]]
    dic_task_graph[name_dict["Graph Spiltter"]] = [ name_dict["Classify1"],name_dict["Classify2"],name_dict["Reco. Merge"] ]
    dic_task_graph[name_dict["Classify1"]] = [name_dict["Reco. Merge"]]
    dic_task_graph[name_dict["Classify2"]] = [name_dict["Reco. Merge"]]
    dic_task_graph[name_dict["Reco. Merge"]] =[name_dict["Display"]]
    dic_task_graph[name_dict["Display"]] = []
    #print("Task Graph Face Recognition\n",dic_task_graph)
    return dic_task_graph

def obj_and_pose_recognition_task_graph():
    name_dict = {"Source":1,
    "Copy":2,
    "Scaler":3,
    "Tiler":4,
    "SIFT1":5,
    "SIFT2":6,
    "SIFT3":7,
    "SIFT4":8,
    "SIFT5":9,
    "Feature merger":10,
    "Descaler":11,
    "Feature spiltter":12,
    "Model matcher1":13,
    "Model matcher2":14,
    "Model matcher3":15,
    "Match joiner":16,
    "Cluster spiltter":17,
    "Clustering1":18,
    "Clustering2":19,
    "Cluster joiner":20,
    "RANSAC":21,
    "Display":22}
    dic_task_graph = dict()
    dic_task_graph[name_dict["Source"]]=[name_dict["Copy"]]
    dic_task_graph[name_dict["Copy"]] = [ name_dict["Scaler"],name_dict["Display"]]
    dic_task_graph[name_dict["Scaler"]] = [ name_dict["Tiler"],name_dict["Descaler"]]
    dic_task_graph[name_dict["Tiler"]] = [ name_dict["SIFT1"],name_dict["SIFT2"],name_dict["SIFT3"],name_dict["SIFT4"],name_dict["SIFT5"],name_dict["Feature merger"] ]
    dic_task_graph[name_dict["SIFT1"]] = [name_dict["Feature merger"]]
    dic_task_graph[name_dict["SIFT2"]] = [name_dict["Feature merger"]]
    dic_task_graph[name_dict["SIFT3"]] = [name_dict["Feature merger"]]
    dic_task_graph[name_dict["SIFT4"]] = [name_dict["Feature merger"]]
    dic_task_graph[name_dict["SIFT5"]] = [name_dict["Feature merger"]]
    dic_task_graph[name_dict["Feature merger"]] = [name_dict["Descaler"]]
    dic_task_graph[name_dict["Descaler"]] = [name_dict["Feature spiltter"]]
    dic_task_graph[name_dict["Feature spiltter"]] = [name_dict["Model matcher1"],name_dict["Model matcher2"],name_dict["Model matcher3"]]
    dic_task_graph[name_dict["Model matcher1"]] = [name_dict["Match joiner"]]
    dic_task_graph[name_dict["Model matcher2"]] = [name_dict["Match joiner"]]
    dic_task_graph[name_dict["Model matcher3"]] = [name_dict["Match joiner"]]
    dic_task_graph[name_dict["Match joiner"]] = [name_dict["Cluster spiltter"]]
    dic_task_graph[name_dict["Cluster spiltter"]] = [name_dict["Clustering1"],name_dict["Clustering2"]]
    dic_task_graph[name_dict["Clustering1"]] = [name_dict["Cluster joiner"]]
    dic_task_graph[name_dict["Clustering2"]] = [name_dict["Cluster joiner"]]
    dic_task_graph[name_dict["Cluster joiner"]] = [name_dict["RANSAC"]]
    dic_task_graph[name_dict["RANSAC"]] =[name_dict["Display"]]
    dic_task_graph[name_dict["Display"]] = []
    #print("Task Graph Obj_and_Pose Recognition\n",dic_task_graph)
    return dic_task_graph
def gesture_recognition_task_graph():
    name_dict = {"Source":1,
    "Copy":2,
    "L_Pair generator":3,
    "L_Scaler":4,
    "L_Tiler":5,
    "L_motionSIFT1":6,
    "L_motionSIFT2":7,
    "L_motionSIFT3":8,
    "L_motionSIFT4":9,
    "L_motionSIFT5":10,
    "L_Feature merger":11,
    "L_Descaler":12,
    "L_Copy":13,
    "R_Scaler":14,
    "R_Tiler":15,
    "R_Face detect1":16,
    "R_Face detect2":17,
    "R_Face detect3":18,
    "R_Face detect4":19,
    "R_Face merger":20,
    "R_Descaler":21,
    "R_Copy":22,
    "Display":23}
    dic_task_graph = dict()
    dic_task_graph[name_dict["Source"]]=[name_dict["Copy"]]
    dic_task_graph[name_dict["Copy"]] = [ name_dict["L_Pair generator"],name_dict["Display"],name_dict["R_Scaler"]]
    dic_task_graph[name_dict["L_Pair generator"]] = [ name_dict["L_Scaler"]]
    dic_task_graph[name_dict["L_Scaler"]] = [ name_dict["L_Tiler"],name_dict["L_Descaler"]]
    dic_task_graph[name_dict["L_Tiler"]] = [ name_dict["L_motionSIFT1"],name_dict["L_motionSIFT2"],name_dict["L_motionSIFT3"],
    name_dict["L_motionSIFT4"],name_dict["L_motionSIFT5"],name_dict["L_Feature merger"] ]
    dic_task_graph[name_dict["L_motionSIFT1"]] = [name_dict["L_Feature merger"]]
    dic_task_graph[name_dict["L_motionSIFT2"]] = [name_dict["L_Feature merger"]]
    dic_task_graph[name_dict["L_motionSIFT3"]] = [name_dict["L_Feature merger"]]
    dic_task_graph[name_dict["L_motionSIFT4"]] = [name_dict["L_Feature merger"]]
    dic_task_graph[name_dict["L_motionSIFT5"]] = [name_dict["L_Feature merger"]]
    dic_task_graph[name_dict["L_Feature merger"]] = [name_dict["L_Descaler"]]
    dic_task_graph[name_dict["L_Descaler"]] = [name_dict["L_Copy"]]
    dic_task_graph[name_dict["L_Copy"]] = [name_dict["Display"]]
    
    dic_task_graph[name_dict["R_Scaler"]] = [ name_dict["R_Tiler"],name_dict["R_Descaler"]]
    dic_task_graph[name_dict["R_Tiler"]] = [ name_dict["R_Face detect1"],name_dict["R_Face detect2"],name_dict["R_Face detect3"],
    name_dict["R_Face detect4"],name_dict["R_Face merger"] ]
    dic_task_graph[name_dict["R_Face detect1"]] = [name_dict["R_Face merger"]]
    dic_task_graph[name_dict["R_Face detect2"]] = [name_dict["R_Face merger"]]
    dic_task_graph[name_dict["R_Face detect3"]] = [name_dict["R_Face merger"]]
    dic_task_graph[name_dict["R_Face detect4"]] = [name_dict["R_Face merger"]]
    dic_task_graph[name_dict["R_Face merger"]] = [name_dict["R_Descaler"]]
    dic_task_graph[name_dict["R_Descaler"]] = [name_dict["R_Copy"]]
    dic_task_graph[name_dict["R_Copy"]] = [name_dict["Display"]]
    dic_task_graph[name_dict["Display"]] = []
    #print("Task Graph Gesture Recognition\n",dic_task_graph)
    return dic_task_graph




def heft_task_assignment(small_DAG,p,d,e,Bandwith):
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
    print("=======================   new running of HEFT ==========")
    # dict_processing_tasks = dict()
    # for i in list(distinct_element_dict(small_DAG)):
    #     x = comp_amount[0,i]
    #     dict_processing_tasks[i]= x.item()
    def commcost(ni, nj, A, B):
        if(A==B):
            return 0
        else:
            #print(f"ni {ni} A {A} {ord(A)-97+1} B {B} {ord(B)-97+1}")
            return d[ni]/Bandwith[ord(A)-97+1][ord(B)-97+1] #average_com_across_nodes
    def compcost(job,agent):
        #global new_P,E
        #nonlocal dict_processing_tasks,speed
        #execution_sp = speed[0,ord(agent)-97]
        x = p[job]/ e[ord(agent)-97+1]   
        return x
    string_naming_machines_with_alphabets = ''
    for i in range(len(e.keys())):
        string_naming_machines_with_alphabets += chr(97+i)

    orders, jobson = schedule(small_DAG, string_naming_machines_with_alphabets, compcost, commcost)
    
    # categorical_labels = [None for _ in range(len(jobson.keys()))]
    # start_index_offset = min(list(small_DAG.keys()))
    # for k,v in jobson.items():
    #     categorical_labels[k-start_index_offset] = ord(v)-97
    # targets = np.array(categorical_labels)
    return orders, jobson





def generate_makespan_problem(task_graph, p, d, e, B, filename):#, rng, max_coef=100):
    """
    Generates a makespan instance with specified characteristics, and writes
    it to a file in the LP format.

    
    Parameters
    ----------
    dag : Dict (Assume starting task is min index of tasks)
        dictionary of dag of task graph (indices start from 1)
    p : Dict
        dictionary of required computations for each task
    e: Dict (indices start from 1)
        dictionary of exec. speed of machines
    B: Dict 
        bandwidth between machines 
    d: Dict
        dictionary of generated data after exec. each task 
    rng: numpy.random.RandomState
        Random number generator
    max_coef: int
        Maximum objective coefficient (>=1)
    """
    num_tasks = len(task_graph.keys())
    drain_tasks = set(
        k for k,v in task_graph.items() if len(v)==0
    )
    #print("drain_tasks",drain_tasks)
    # ------------- generate random settings -----------
    # d = {k:np.random.rand() for k in task_graph.keys()}
    # p = {k:np.random.rand() for k in task_graph.keys()}
    
    # num_machines = 4
    # e = {k:np.random.uniform(low=.3, high=1.0) for k in range(1,num_machines+1)}
    # B_list = [[np.random.uniform(low=.3, high=1.0) for _ in range(num_machines)] for _ in range(num_machines)]#

    # for i in range(num_machines):
    #     B_list[i][i] = 10.0    
    # e = {1:1.0,2:1.0}
    # B_list = [[1,1],[2,2]]
    # --------------------------------------------------
    
    # B = dict()
    # for elem in e.keys():
    #     B[elem] = dict()
    #     for succ_elem in e.keys():
    #         B[elem][succ_elem] = B_list[elem-1][succ_elem-1]
    # --------------------------------------------------
    c_t = [1.0]
    M = 1000
    upper_bound_s = sum([v for v in p.values()])/(1.0*max([speed for speed in e.values()]))
    if upper_bound_s >= 0.1*M:
        raise "upper bound is close to M!!! "
    # ----- START:create preceeding matrix indicator ------
    Q = dict()
    for elem in task_graph.keys():
        Q[elem] = dict()
        for succ_elem in task_graph.keys():
            Q[elem][succ_elem] = 0 
    #Q = [[0 for j in range(num_tasks)] for i in range(num_tasks)]
    #print(f"Taks graph:\n")
    #for k,v in task_graph.items():
    #    print(f"{k}{v}")
    def dfs_search(n):
        nonlocal set_sofar
        #print(f"n {n},sofar  {set_sofar}")
            
        for element in set_sofar:
            Q[element][n] = 1
            Q[n][element] = 1
        
        set_sofar.add(n)
        for child in task_graph[n]:
            dfs_search(child)
        set_sofar.remove(n)
    
    set_sofar = set() #set([min(task_graph.keys())])
    dfs_search(min(task_graph.keys()))
    #print(f"Q :{Q}")
     
    # ----- END:create preceeding matrix indicator ------
    
    # write problem
    with open(filename, 'w') as file:
        
        file.write("minimize\nOBJ:")
        file.write("".join([f" +1 t_{j+1}" for j in range(len(c_t))]))
        
        
        file.write("\n\nsubject to\n")
        ## for enforcing t_1 >= s_j + \sum_m x[j][m]*p[j]/e[m] for all j,m 
        idx_constraint = 0
        for j in task_graph.keys():
            row_cols_str = "".join([f" +1 t_1 -1 s_{j}"]) + "".join([f" -{p[j]/e[m]} x_{j}_{m}" for m in e.keys()])
            file.write(f"C{idx_constraint}:" + row_cols_str + f" >= 0\n")
            idx_constraint +=1
        
        ## for enforcing s_j >= 0 for all j in task_graph
        # for j in task_graph.keys():
        #     row_cols_str = "".join([f" +1 s_{j}"])
        #     file.write(f"C{idx_constraint}:" + row_cols_str + f" >= 0\n")
        #     idx_constraint += 1

        ## for enforcing s_j >= 0 for all j in task_graph
        for j in task_graph.keys():
            row_cols_str = "".join([f" +1 x_{j}_{m}" for m in e.keys()])
            file.write(f"C{idx_constraint}:" + row_cols_str + f" == 1\n")
            idx_constraint += 1

        ## for enforcing s_j >= 0 for all j in task_graph
        for j in task_graph.keys():
            for k in task_graph[j]:
                row_cols_str = "".join([f" +1 s_{k} -1 s_{j}"])
                file.write(f"C{idx_constraint}:" + row_cols_str + f" >= 0\n")
                idx_constraint += 1

                for a in e.keys():
                    for b in e.keys():
                        #print(f"a{a} b{b} j{j}")
                        D_j_a_PLUS_C_j_a_b = (d[j]/B[a][b])+(p[j]/e[a])
                        row_cols_str = "".join([f" +1 s_{k} -1 s_{j} -{D_j_a_PLUS_C_j_a_b} x_{j}_{a} -{D_j_a_PLUS_C_j_a_b} x_{k}_{b}"])#+ {(d[j]/B[m][n])+p[ell]/e[n]}
                        file.write(f"C{idx_constraint}:" + row_cols_str + f" >= -{D_j_a_PLUS_C_j_a_b}\n")
                        idx_constraint += 1

        
        ## for enforcing nonoverlapping tasks
        for j in task_graph.keys():
            for k in task_graph.keys():
                if j != k and Q[j][k]==0:
                    #print(f"could overlap:{j},{k}")
                    row_cols_str = "".join([f" +1 s_{k} -1 s_{j}"]) + "".join([f" {-p[j]/e[a]} x_{j}_{a}" for a in e.keys()]) + "".join([f" +{M} z_{j}_{k}"])
                    file.write(f"C{idx_constraint}:" + row_cols_str + f" >= 0\n")
                    idx_constraint += 1

                    row_cols_str = "".join([f" +1 s_{k} -1 s_{j}"]) + "".join([f" {-p[j]/e[a]} x_{j}_{a}" for a in e.keys()]) + "".join([f" +{M} z_{j}_{k}"])
                    file.write(f"C{idx_constraint}:" + row_cols_str + f" <= {M-1}\n")
                    idx_constraint += 1

        ## constraint for not overlapping >=3
        for j in task_graph.keys():
            for k in task_graph.keys():
                if j != k:
                    for a in e.keys():
                        row_cols_str = "".join([f" +1 x_{j}_{a} +1 x_{k}_{a} +1 z_{j}_{k} +1 z_{k}_{j}"]) 
                        file.write(f"C{idx_constraint}:" + row_cols_str + f" <= 3\n")
                        idx_constraint += 1


        file.write("\nbounds\n")
        #file.write(f"0 <= t_1\n")
        for j in task_graph.keys():
            file.write(f"0 <= s_{j} <= {M}\n")
        # specify the Binarry variables
        file.write("\nbinary\n")
        file.write("".join([f" x_{j}_{m}" for j in task_graph.keys() for m in e.keys()]) + "".join([f" z_{j}_{k}" for j in task_graph.keys() for k in task_graph.keys() if j != k ])  )


        # file.write("maximize\nOBJ:")
        # file.write("".join([f" +10 x_1 +15 x_2"]))

        # file.write("\n\nsubject to\n")
        # file.write("".join([f" +2 x_1 +3 x_2 <= 5\n"]))
        # file.write("".join([f" +4 x_1 +2 x_2 <= 7\n"]))
        # # file.write("".join([f" +1 x_1 +1 x_2 <= 200\n"]))
        # # file.write("".join([f" -1 x_1 -1 x_2 <= -125\n"]))
        # # file.write("".join([f" +1 x_3 <= 200\n"]))

        # # file.write("\nbounds\n")
        # # file.write(f"0 <= x_1 <= 1\n")
        # # file.write(f"0 <= x_2 <= 1\n")

        # file.write("\nbinary\n")
        # file.write("".join([f" x_1 x_2"]))

if __name__=='__main__':
    path = os.getcwd()
    print(path)
    with gzip.open(path+'/data/samples/indset/500_4/train/sample_1.pkl', 'rb') as f:
    # Load the object from the file
        data = pickle.load(f)

    # Do something with the loaded data
    print(data)




def make_DAG(num_tasks,prob_edge):
    """
    Create single source single destination DAG
    """
    G_making=nx.gnp_random_graph(num_tasks,prob_edge,directed=True)
    G_directed = nx.DiGraph([(u,v) for (u,v) in G_making.edges() if u<v])
    
    set_out_going_nodes,set_in_going_nodes = set(),set() 
    for (u,v) in G_directed.edges():
         set_out_going_nodes.add(u)
         set_in_going_nodes.add(v)
    min_node_idx,max_node_idx = min(G_directed.nodes()),max(G_directed.nodes())     
    sources = set_out_going_nodes.difference(set_in_going_nodes)     
    destinations = set_in_going_nodes.difference(set_out_going_nodes)
    # if there are more than ONE source
    if len(sources)>1:
        for n in sources:
            G_directed.add_edge(min_node_idx-1, n)
    # if there are more than ONE destination
    if len(destinations)>1:
        for n in destinations:
            G_directed.add_edge(n,max_node_idx+1)

    # if FINAL source id is <1 ===> shift it to start from 0
    G_out = nx.DiGraph()
    if len(sources)>1 and min_node_idx==0:
        for (u,v) in G_directed.edges():
            G_out.add_edge(u+1,v+1)
    else:
        G_out = G_directed    

    if not nx.is_directed_acyclic_graph(G_out):
        raise NameError('it is NOT DAG!!!')
    #if not nx.is_directed_acyclic_graph(G_out):
    #    raise NameError('it is NOT DAG!!!')


    dag_start_from_0 = dict()
    for (u,v) in sorted(G_out.edges(),key=lambda x:[x[0],x[1]]):
        if u not in dag_start_from_0.keys():
            dag_start_from_0[u] = [v]
        else:
            dag_start_from_0[u].append(v)    
    
    # to add task_graph[sink_nodes] = []
    if not any(len(v)==0 for v in dag_start_from_0.values() ):
        dest_nodes,source_nodes = set(), set()
        for k,v in  dag_start_from_0.items():
            source_nodes.add(k)
            for elem in v:
                dest_nodes.add(elem)
        sink_nodes = dest_nodes - source_nodes
        for elem in sink_nodes:
            dag_start_from_0[elem] = []


    return dag_start_from_0
    
def make_DAG_MODIFIED(num_tasks,min_deg,max_deg,min_width,max_width,depth,num_of_ahead_layers_considered_for_link):
    """
    num_of_ahead_layers_considered_for_link (int) >= 1
    """
    #max_width
    max_width = min_width = int((num_tasks-1)*1.0/depth)

    nodes_has_incoming_edge = set()
    nodes_has_outcoming_edge = set()
    dict_layer_nodes = {0:[0]}
    index_starting_node = 0
    for i in range(1,depth):
        num_nodes_in_this_layer = random.randrange(min_width,max_width+1)
        dict_layer_nodes[i] = [k for k in range(index_starting_node+1,index_starting_node+num_nodes_in_this_layer+1)]
        index_starting_node += num_nodes_in_this_layer
    
    if index_starting_node+1 <num_tasks:
        dict_layer_nodes[depth] = [k for k in range(index_starting_node+1,num_tasks)]
        depth += 1
    dict_layer_nodes[depth] = [num_tasks]
    source,destination = 0, num_tasks
    #print("Layers\n",dict_layer_nodes[max([k for k in dict_layer_nodes.keys()])][-1]+1, dict_layer_nodes)    

    dic_task_graph = dict()
    dic_task_graph[0] = dict_layer_nodes[1] # connect source to all layer-1 nodes
    nodes_has_outcoming_edge.add(0)
    nodes_has_incoming_edge.update(dict_layer_nodes[1])
    
    # connect destination to all nodes in previous nodes
    for node in  dict_layer_nodes[depth-1]:
        if node not in dic_task_graph.keys():
            dic_task_graph[node] = [destination]
        else:
            dic_task_graph[node].append(destination)
        
        nodes_has_outcoming_edge.add(node)
    nodes_has_incoming_edge.add(destination)

    for k in range(1,max(dict_layer_nodes.keys())-num_of_ahead_layers_considered_for_link):
        
        for v in dict_layer_nodes[k]:
            num_child = random.randrange(min_deg,max_deg+1)
            candidate_nodes_to_set_link = [x for x in range(dict_layer_nodes[k+1][0],dict_layer_nodes[k+num_of_ahead_layers_considered_for_link][-1])]
            #print("num_child ",num_child, " candidate_nodes_to_set_link ", candidate_nodes_to_set_link)
            dic_task_graph[v] = random.sample(candidate_nodes_to_set_link,min(num_child,len(candidate_nodes_to_set_link)))
            
            nodes_has_outcoming_edge.add(v)
            nodes_has_incoming_edge.update(dic_task_graph[v])
            #print("node ",v," children ",dic_task_graph[v])
    
    # make a connection between any node that has no incoming edge (then add a connection just to previous layer)    
    for l in range(1,depth):
        for node in dict_layer_nodes[l]:
            if node not in nodes_has_incoming_edge:
                rand_node_prev_layer = random.randint(dict_layer_nodes[l-1][0],dict_layer_nodes[l-1][-1])
                if rand_node_prev_layer not in dic_task_graph.keys():
                    dic_task_graph[rand_node_prev_layer]=[node]
                else:    
                    dic_task_graph[rand_node_prev_layer].append(node)

                nodes_has_outcoming_edge.add(rand_node_prev_layer)  # update both node and the node from prev layer that a link established between them
                nodes_has_incoming_edge.add(node)

    # make a connection between any node that has no incoming edge (then add a connection just to previous layer)    
    for l in range(depth-1,0,-1):
        for node in dict_layer_nodes[l]:
            if node not in nodes_has_outcoming_edge:
                rand_node_next_layer = random.randint(dict_layer_nodes[l+1][0],dict_layer_nodes[l+1][-1])
                if node not in dic_task_graph.keys():
                    dic_task_graph[node]=[rand_node_next_layer]
                else:
                    dic_task_graph[node].append(rand_node_next_layer)

                nodes_has_incoming_edge.add(rand_node_next_layer)  # update both node and the node from prev layer that a link established between them
                nodes_has_outcoming_edge.add(node)
    if len(nodes_has_incoming_edge)!=len(nodes_has_outcoming_edge) or len(nodes_has_incoming_edge)!=num_tasks or len(nodes_has_outcoming_edge)!=num_tasks:
        raise NameError('it is NOT correct!!!') 
    # for k in sorted(dic_task_graph.keys()):
    #     print("=",k,dic_task_graph[k]) 
    return dic_task_graph