import networkx as nx
import numpy as np
import itertools as it
import datetime as dt


#########################################################################################
############################### Network Helper Functions  ###############################
#########################################################################################
print("Hello this has loaded", dt.datetime.now())

def set_time(G, value, node=None, time='time'):
    '''
        Set the time of an individual node in a network G 
        to value or set the time of all nodes to value.
        G      ::  a networkx graph
        node   ::  a reference to a node in G
        value  ::  a non-negative integer
    '''
    if node:
        G.nodes[node][time] = value
    else:
        time_attrib = {i : value for i in G.nodes()}
        nx.set_node_attributes(G,time_attrib, time)


def set_influence(G, value, node=None, label='is_influenced'):
    '''
        Set the influence of an individual node in a network G 
        to value or set the influence of all nodes to value.
        G      ::  a networkx graph
        node   ::  a reference to a node in G
        value  ::  an integer 0 or 1
    '''
    if node:
        G.nodes[node][label] = value
    else:
        influence_attrib = { i : value for i in G.nodes() }
        nx.set_node_attributes(G,influence_attrib, label)
        
def get_is_influenced(G, node, label='is_influenced'):
    '''
        Returns if node in G is influenced.
    '''
    return G.nodes[node][label]
        
def get_number_influenced(G, label='is_influenced'):
    '''
        Get the number of influenced nodes.
    '''
    return sum(nx.get_node_attributes(G, label).values())

#################################################################################
########################## Simulation Helper Functions ##########################
#################################################################################

def get_uninfluenced_neighbours(G, nodes, label='is_influenced'):
    '''
        Return a set of neighbours of nodes
        that are uninfluenced.
    '''
    neighbours = set()
    for node in nodes:
        friends = list(G.neighbors(node))
        neighbours.update([friend for friend in friends if G.nodes[friend][label] == 0])
        ## implication is no node added is in nodes because nodes should all be influenced 
    
    ## In case the above implication doesn't hold...
#     tmp = set(nodes)
#     neighbours = neighbours - tmp
    return neighbours

def update_influence(G, node, phi, time, label='is_influenced'):
    '''
        Assumes the node isn't currently influenced.
        Update a node's influence status.
        Returns true or false.
    '''
    friends = list(G.neighbors(node))
    num_friends = len(friends)

    ## Node with no friends cannot be influenced
    if num_friends == 0:
        return False

    ## Calculate the number of friends who can influence 
    ## current node and compare with threshold.
    num_influenced = sum([1 for friend in friends if G.nodes[friend][label] == 1])
    if (num_influenced/num_friends) > phi:
        set_influence(G, 1, node=node)
        set_time(G, time, node=node)
        return True
    return False
    
def simulate_spread(G, initial_node, phi):
    '''
        Simulates the spread of influence from initial node under threshold phi.
        Tracks the component of influenced nodes, determines the uninfluenced 
        neighbours of this component, and determines whether the neighbours 
        can be influenced. 
        Returns the number of influenced nodes and expected time to be influenced.
    '''
    
    G_tmp = G.copy()
    set_influence(G_tmp, 1, node=initial_node)
    N = G_tmp.number_of_nodes()
    t = [0 for _ in range(N)]
    time, num_influenced = 1, 1
    t[0] = 1
    influenced_nodes = set([initial_node])
    
    ## Iteratively compute the number of nodes (update t[time]) influenced at
    ## each time step until a time step is reached where no neighbours to
    ## the influenced component can be influenced.
    while num_influenced > 0:
        num_influenced = 0
        neighbours = get_uninfluenced_neighbours(G_tmp, influenced_nodes)
        for node in neighbours:
            if update_influence(G_tmp, node, phi, time):
                num_influenced += 1
                influenced_nodes.add(node)
        t[time] = num_influenced
        time += 1
    
    ## Determine the empirical expected time to be influenced
    expected_time = sum([i * t[i] for i in range(N)])/N
    return (len(influenced_nodes), expected_time)


def simulate_spread_wrapper(G, node_list, phi):
    for node in node_list:
        set_influence(G, 1, node=node)
    return simulate_spread_generic(G,phi)

def simulate_spread_generic(G, phi):
    N = G.number_of_nodes()
    t = [0 for _ in range(N)]
    time, num_influenced = 1, 1
    t[0] = 1
    influenced_nodes = set([node for node in G.nodes() if get_is_influenced(G, node) == 1 ])
    
    ## Iteratively compute the number of nodes (update t[time]) influenced at
    ## each time step until a time step is reached where no neighbours to
    ## the influenced component can be influenced.
    while num_influenced > 0:
        num_influenced = 0
        neighbours = get_uninfluenced_neighbours(G, influenced_nodes)
        for node in neighbours:
            if update_influence(G, node, phi, time):
                num_influenced += 1
                influenced_nodes.add(node)
        t[time] = num_influenced
        time += 1
    
    ## Determine the empirical expected time to be influenced
    expected_time = sum([i * t[i] for i in range(N)])/N
    return (len(influenced_nodes), expected_time)

def update_influence_directed(G, node, phi, time, label='is_influenced'):
    '''
        Assumes the node isn't currently influenced.
        Update a node's influence status.
        Returns true or false.
    '''
    friends = [x for x, _ in G.in_edges(node)]
    num_friends = len(friends)

    ## Node with no friends cannot be influenced
    if num_friends == 0:
        return False

    ## Calculate the number of friends who can influence 
    ## current node and compare with threshold.
    num_influenced = sum([1 for friend in friends if G.nodes[friend][label] == 1])
    if (num_influenced/num_friends) > phi:
        set_influence(G, 1, node=node)
        set_time(G, time, node=node)
        return True
    return False
    
def simulate_spread_directed(G, initial_node, phi):
    '''
        Simulates the spread of influence from initial node under threshold phi.
        Tracks the component of influenced nodes, determines the uninfluenced 
        neighbours of this component, and determines whether the neighbours 
        can be influenced. 
        Returns the number of influenced nodes and expected time to be influenced.
    '''
    
    G_tmp = G.copy()
    set_influence(G_tmp, 1, node=initial_node)
    N = G_tmp.number_of_nodes()
    t = [0 for _ in range(N)]
    time, num_influenced = 1, 1
    t[0] = 1
    influenced_nodes = set([initial_node])
    
    ## Iteratively compute the number of nodes (update t[time]) influenced at
    ## each time step until a time step is reached where no neighbours to
    ## the influenced component can be influenced.
    while num_influenced > 0:
        num_influenced = 0
        neighbours = get_uninfluenced_neighbours(G_tmp, influenced_nodes)
        for node in neighbours:
            if update_influence_directed(G_tmp, node, phi, time):
                num_influenced += 1
                influenced_nodes.add(node)
        t[time] = num_influenced
        time += 1
    
    ## Determine the empirical expected time to be influenced
    expected_time = sum([i * t[i] for i in range(N)])/N
    return (len(influenced_nodes), expected_time)


#################################################################################
############################# Simulation  Functions #############################
#################################################################################

def run_simulation(G, phi=0.18, q=0.1, directed=False):
    '''
        Simulation of influential cascade on network G.
        Returns the average size of influenced nodes and average expected 
        time to be influenced from 
            - influential nodes
            - normal nodes
    '''
    set_influence(G, 0)
    set_time(G, 0)
    
    degree_ordered_nodes = sorted(list(G.nodes()), key=lambda x: G.degree(x), reverse=True)
    N = G.number_of_nodes()
    degree_ordered_nodes = sorted(list(G.nodes()), key=lambda x: G.degree(x), reverse=True)
    influential_nodes_5   = degree_ordered_nodes[:int(0.05*N)]
    influential_nodes_10  = degree_ordered_nodes[int(0.05*N):int(0.1*N)]
    influential_nodes_15 = degree_ordered_nodes[int(0.1*N):int(0.15*N)]
    influential_nodes_20 = degree_ordered_nodes[int(0.15*N):int(0.2*N)]
    bottom_nodes = degree_ordered_nodes[int(0.9*N):]
    
    average = np.mean(list(dict(G.degree()).values()))
    lower, upper = int(np.floor(average)), int(np.ceil(average))
    normal_nodes = [x for x in G.nodes() if lower <= G.degree(x) <= upper ]
    
    influential_S_5, influential_S_10, influential_S_15, influential_S_20 = [], [], [], []
    influential_t_5, influential_t_10, influential_t_15, influential_t_20 = [], [], [], []
    normal_S, bottom_S = [], []
    normal_t, bottom_t = [], []
    
    ## Calculate the number of influenced nodes (S) and expected time of influenced nodes
    ## for each influential node
    for node in influential_nodes_5:
        S, t = simulate_spread(G, node, phi)
        influential_S_5.append(S)
        influential_t_5.append(t)    
    for node in influential_nodes_10:
        S, t = simulate_spread(G, node, phi)
        influential_S_10.append(S)
        influential_t_10.append(t)
    for node in influential_nodes_15:
        S, t = simulate_spread(G, node, phi)
        influential_S_15.append(S)
        influential_t_15.append(t)
    for node in influential_nodes_20:
        S, t = simulate_spread(G, node, phi)
        influential_S_20.append(S)
        influential_t_20.append(t)
    for node in bottom_nodes:
        S, t = simulate_spread(G, node, phi)
        bottom_S.append(S)
        bottom_t.append(t)
    for node in normal_nodes:
        S, t = simulate_spread(G, node, phi)
        normal_S.append(S)
        normal_t.append(t)

    return [np.mean(influential_S_5), np.mean(influential_S_10), np.mean(influential_S_15), np.mean(influential_S_20), np.mean(influential_S_5 + influential_S_10), np.mean(influential_S_5 + influential_S_10 + influential_S_15), np.mean(influential_S_5 + influential_S_10 + influential_S_15 + influential_S_20), np.mean(normal_S), np.mean(bottom_S),
            np.mean(influential_t_5), np.mean(influential_t_10), np.mean(influential_t_15), np.mean(influential_t_20), np.mean(influential_t_5 + influential_t_10), np.mean(influential_t_5 + influential_t_10 + influential_t_15), np.mean(influential_t_5 + influential_t_10 + influential_t_15 + influential_t_20), np.mean(normal_t), np.mean(bottom_t)]

def run_simulation_RG_directed(N,p,phi=0.18):
    '''
        Simulation of Poisson/Binomial Random Graph
        Returns the average size of influenced nodes and average expected 
        time to be influenced from 
            - influential nodes
            - normal nodes
    '''
    G = nx.erdos_renyi_graph(N,p,directed=True)
    set_influence(G, 0)
    set_time(G, 0)
    
    ## Retrieve influential nodes - top q% and non-influential nodes
    degree_ordered_nodes = sorted(list(G.nodes()), key=lambda x: G.degree(x), reverse=True)
    influential_nodes_5   = degree_ordered_nodes[:int(0.05*N)]
    influential_nodes_10  = degree_ordered_nodes[int(0.05*N):int(0.1*N)]
    influential_nodes_15 = degree_ordered_nodes[int(0.1*N):int(0.15*N)]
    influential_nodes_20 = degree_ordered_nodes[int(0.15*N):int(0.2*N)]
    bottom_nodes = degree_ordered_nodes[int(0.9*N):]
        
    average = p * (N-1)
    lower = np.mean(list(dict(G.degree()).values())) - np.std(list(dict(G.degree()).values()))/2
    upper = np.mean(list(dict(G.degree()).values())) + np.std(list(dict(G.degree()).values()))/2
    normal_nodes = [x for x in G.nodes() if lower <= G.degree(x) <= upper ]
    
    influential_S_5, influential_S_10, influential_S_15, influential_S_20 = [], [], [], []
    influential_t_5, influential_t_10, influential_t_15, influential_t_20 = [], [], [], []
    normal_S, bottom_S = [], []
    normal_t, bottom_t = [], []
    
    ## Calculate the number of influenced nodes (S) and expected time of influenced nodes
    ## for each influential node
    for node in influential_nodes_5:
        S, t = simulate_spread_directed(G, node, phi)
        influential_S_5.append(S)
        influential_t_5.append(t)    
    for node in influential_nodes_10:
        S, t = simulate_spread_directed(G, node, phi)
        influential_S_10.append(S)
        influential_t_10.append(t)
    for node in influential_nodes_15:
        S, t = simulate_spread_directed(G, node, phi)
        influential_S_15.append(S)
        influential_t_15.append(t)
    for node in influential_nodes_20:
        S, t = simulate_spread_directed(G, node, phi)
        influential_S_20.append(S)
        influential_t_20.append(t)
    for node in bottom_nodes:
        S, t = simulate_spread_directed(G, node, phi)
        bottom_S.append(S)
        bottom_t.append(t)
    for node in normal_nodes:
        S, t = simulate_spread_directed(G, node, phi)
        normal_S.append(S)
        normal_t.append(t)
    
    return [np.mean(influential_S_5), np.mean(influential_S_10), np.mean(influential_S_15), np.mean(influential_S_20), np.mean(influential_S_5 + influential_S_10), np.mean(influential_S_5 + influential_S_10 + influential_S_15), np.mean(influential_S_5 + influential_S_10 + influential_S_15 + influential_S_20), np.mean(normal_S), np.mean(bottom_S),
            np.mean(influential_t_5), np.mean(influential_t_10), np.mean(influential_t_15), np.mean(influential_t_20), np.mean(influential_t_5 + influential_t_10), np.mean(influential_t_5 + influential_t_10 + influential_t_15), np.mean(influential_t_5 + influential_t_10 + influential_t_15 + influential_t_20), np.mean(normal_t), np.mean(bottom_t)]



def run_simulation_RG(N,p,phi=0.18):
    '''
        Simulation of Poisson/Binomial Random Graph
        Returns the average size of influenced nodes and average expected 
        time to be influenced from 
            - influential nodes
            - normal nodes
    '''
    G = nx.erdos_renyi_graph(N,p)
    set_influence(G, 0)
    set_time(G, 0)
    
    ## Retrieve influential nodes - top q% and non-influential nodes
    degree_ordered_nodes = sorted(list(G.nodes()), key=lambda x: G.degree(x), reverse=True)
    influential_nodes_5   = degree_ordered_nodes[:int(0.05*N)]
    influential_nodes_10  = degree_ordered_nodes[int(0.05*N):int(0.1*N)]
    influential_nodes_15 = degree_ordered_nodes[int(0.1*N):int(0.15*N)]
    influential_nodes_20 = degree_ordered_nodes[int(0.15*N):int(0.2*N)]
    bottom_nodes = degree_ordered_nodes[int(0.9*N):]
        
    average = p * (N-1)
    lower, upper = int(np.floor(average)), int(np.ceil(average))
    normal_nodes = [x for x in G.nodes() if lower <= G.degree(x) <= upper ]

    influential_S_5, influential_S_10, influential_S_15, influential_S_20 = [], [], [], []
    influential_t_5, influential_t_10, influential_t_15, influential_t_20 = [], [], [], []
    normal_S, bottom_S = [], []
    normal_t, bottom_t = [], []
    
    ## Calculate the number of influenced nodes (S) and expected time of influenced nodes
    ## for each influential node
    for node in influential_nodes_5:
        S, t = simulate_spread(G, node, phi)
        influential_S_5.append(S)
        influential_t_5.append(t)    
    for node in influential_nodes_10:
        S, t = simulate_spread(G, node, phi)
        influential_S_10.append(S)
        influential_t_10.append(t)
    for node in influential_nodes_15:
        S, t = simulate_spread(G, node, phi)
        influential_S_15.append(S)
        influential_t_15.append(t)
    for node in influential_nodes_20:
        S, t = simulate_spread(G, node, phi)
        influential_S_20.append(S)
        influential_t_20.append(t)
    for node in bottom_nodes:
        S, t = simulate_spread(G, node, phi)
        bottom_S.append(S)
        bottom_t.append(t)
    for node in normal_nodes:
        S, t = simulate_spread(G, node, phi)
        normal_S.append(S)
        normal_t.append(t)
    
    return [np.mean(influential_S_5), np.mean(influential_S_10), np.mean(influential_S_15), np.mean(influential_S_20), np.mean(influential_S_5 + influential_S_10), np.mean(influential_S_5 + influential_S_10 + influential_S_15), np.mean(influential_S_5 + influential_S_10 + influential_S_15 + influential_S_20), np.mean(normal_S), np.mean(bottom_S),
            np.mean(influential_t_5), np.mean(influential_t_10), np.mean(influential_t_15), np.mean(influential_t_20), np.mean(influential_t_5 + influential_t_10), np.mean(influential_t_5 + influential_t_10 + influential_t_15), np.mean(influential_t_5 + influential_t_10 + influential_t_15 + influential_t_20), np.mean(normal_t), np.mean(bottom_t)]


def run_simulation_SF(N,n_avg,phi=0.18):
    '''
        Simulation of Scale Free Random Graphs
        Returns the average size of influenced nodes and average expected 
        time to be influenced from 
            - influential nodes
            - normal nodes
        NOTE. n_avg must be an integer.
    '''
    G = nx.barabasi_albert_graph(N,n_avg)
    set_influence(G, 0)
    set_time(G, 0)
    
    ## Retrieve influential nodes - top q% and non-influential nodes
    degree_ordered_nodes = sorted(list(G.nodes()), key=lambda x: G.degree(x), reverse=True)
    influential_nodes_5   = degree_ordered_nodes[:int(0.05*N)]
    influential_nodes_10  = degree_ordered_nodes[int(0.05*N):int(0.1*N)]
    influential_nodes_15 = degree_ordered_nodes[int(0.1*N):int(0.15*N)]
    influential_nodes_20 = degree_ordered_nodes[int(0.15*N):int(0.2*N)]
    bottom_nodes = degree_ordered_nodes[int(0.9*N):]
    
    ## Normal nodes are nodes with degree close to average
    lower, upper = int(np.floor(n_avg)), int(np.ceil(n_avg))
    normal_nodes = [x for x in G.nodes() if lower <= G.degree(x) <= upper ]

    influential_S_5, influential_S_10, influential_S_15, influential_S_20 = [], [], [], []
    influential_t_5, influential_t_10, influential_t_15, influential_t_20 = [], [], [], []
    normal_S, bottom_S = [], []
    normal_t, bottom_t = [], []
    
    ## Calculate the number of influenced nodes (S) and expected time of influenced nodes
    ## for each influential node
    for node in influential_nodes_5:
        S, t = simulate_spread(G, node, phi)
        influential_S_5.append(S)
        influential_t_5.append(t)    
    for node in influential_nodes_10:
        S, t = simulate_spread(G, node, phi)
        influential_S_10.append(S)
        influential_t_10.append(t)
    for node in influential_nodes_15:
        S, t = simulate_spread(G, node, phi)
        influential_S_15.append(S)
        influential_t_15.append(t)
    for node in influential_nodes_20:
        S, t = simulate_spread(G, node, phi)
        influential_S_20.append(S)
        influential_t_20.append(t)
    for node in bottom_nodes:
        S, t = simulate_spread(G, node, phi)
        bottom_S.append(S)
        bottom_t.append(t)
    for node in normal_nodes:
        S, t = simulate_spread(G, node, phi)
        normal_S.append(S)
        normal_t.append(t)
    
    return [np.mean(influential_S_5), np.mean(influential_S_10), np.mean(influential_S_15), np.mean(influential_S_20), np.mean(influential_S_5 + influential_S_10), np.mean(influential_S_5 + influential_S_10 + influential_S_15), np.mean(influential_S_5 + influential_S_10 + influential_S_15 + influential_S_20), np.mean(normal_S), np.mean(bottom_S),
            np.mean(influential_t_5), np.mean(influential_t_10), np.mean(influential_t_15), np.mean(influential_t_20), np.mean(influential_t_5 + influential_t_10), np.mean(influential_t_5 + influential_t_10 + influential_t_15), np.mean(influential_t_5 + influential_t_10 + influential_t_15 + influential_t_20), np.mean(normal_t), np.mean(bottom_t)]



def run_simulation_RG_PC(N,p,phi=0.18, groupsize = 2):
    '''
        Simulation of Poisson/Binomial Random Graph
        Returns the average size of influenced nodes and average expected 
        time to be influenced from 
            - influential nodes
            - normal nodes
    '''
    G = nx.erdos_renyi_graph(N,p)
    set_influence(G, 0)
    set_time(G, 0)
    
    ## Retrieve influential nodes - top q% and non-influential nodes
    degree_ordered_nodes = sorted(list(G.nodes()), key=lambda x: G.degree(x), reverse=True)
    influential_nodes_5   = degree_ordered_nodes[:int(0.05*N)]
    influential_nodes_10  = degree_ordered_nodes[int(0.05*N):int(0.1*N)]
    influential_nodes_15 = degree_ordered_nodes[int(0.1*N):int(0.15*N)]
    influential_nodes_20 = degree_ordered_nodes[int(0.15*N):int(0.2*N)]
    bottom_nodes = degree_ordered_nodes[int(0.9*N):]
        
    average = p * (N-1)
    lower, upper = int(np.floor(average)), int(np.ceil(average))
    normal_nodes = [x for x in G.nodes() if lower <= G.degree(x) <= upper ]

    influential_S_5, influential_S_10, influential_S_15, influential_S_20 = [], [], [], []
    influential_t_5, influential_t_10, influential_t_15, influential_t_20 = [], [], [], []
    normal_S, bottom_S = [], []
    normal_t, bottom_t = [], []
    
    ## Calculate the number of influenced nodes (S) and expected time of influenced nodes
    ## for each influential node
    
    ## Generate group for influential nodes 0-5 and groups of 4 for influential_nodes_10
    if groupsize <= np.floor(N/40) + 1:
        influential_nodes_5_group = it.combinations(influential_nodes_5, groupsize)
        influential_nodes_10_group = it.combinations(influential_nodes_10, groupsize)
        influential_nodes_15_group = it.combinations(influential_nodes_15, groupsize)
        influential_nodes_20_group = it.combinations(influential_nodes_20, groupsize)
        normal_nodes_group = it.combinations(normal_nodes, groupsize)
        bottom_nodes_group = it.combinations(bottom_nodes, groupsize)
    ## else:
    ##     nodes_copy = list(G.nodes())
    ##     notgroupsize = int(N/20 - groupsize)
    ##     influential_nodes_5_group = it.combinations(influential_nodes_5, notgroupsize)
    ##     influential_nodes_5_group = np.setdiff1d(nodes_copy,influential_nodes_5_group)
    ##     influential_nodes_10_group = it.combinations(influential_nodes_10, notgroupsize)
    ##     influential_nodes_10_group = np.setdiff1d(nodes_copy,influential_nodes_10_group)
    ##     influential_nodes_15_group = it.combinations(influential_nodes_15, notgroupsize)
    ##     influential_nodes_15_group = np.setdiff1d(nodes_copy,influential_nodes_15_group)
    ##     influential_nodes_20_group = it.combinations(influential_nodes_20, notgroupsize)
    ##     influential_nodes_20_group = np.setdiff1d(nodes_copy,influential_nodes_20_group)
    ##     normal_nodes_group = it.combinations(normal_nodes, notgroupsize)
    ##     normal_nodes_group = np.setdiff1d(nodes_copy,normal_nodes_group)
    ##     bottom_nodes_group = it.combinations(bottom_nodes, notgroupsize)
    ##     bottom_nodes_group = np.setdiff1d(nodes_copy,bottom_nodes_group)
    elif groupsize == 4:
        influential_nodes_5_group = []
        influential_nodes_10_group = []
        influential_nodes_15_group = []
        influential_nodes_20_group = []
        normal_nodes_group = []
        bottom_nodes_group = []
        for i in np.arange(5):
            #influential_nodes_5_group.append([x for x in influential_nodes_5 if x != influential_nodes_5[i]])
            #influential_nodes_10_group.append([x for x in influential_nodes_10 if x != influential_nodes_10[i]])
            #influential_nodes_15_group.append([x for x in influential_nodes_15 if x != influential_nodes_15[i]])
            #influential_nodes_20_group.append([x for x in influential_nodes_20 if x != influential_nodes_20[i]])
            #normal_nodes_group.append([x for x in normal_nodes if x != normal_nodes[i]])
            #bottom_nodes_group.append([x for x in bottom_nodes if x != bottom_nodes[i]])
            
            nodes_05 = np.array(influential_nodes_5)
            nodes_10 = np.array(influential_nodes_10)
            nodes_15 = np.array(influential_nodes_15)
            nodes_20 = np.array(influential_nodes_20)
            normal = np.array(normal_nodes)
            bottom = np.array(bottom_nodes)
            
            influential_nodes_5_group.append(nodes_05[np.arange(len(nodes_05))!=i].tolist())
            influential_nodes_10_group.append(nodes_10[np.arange(len(nodes_10))!=i].tolist())
            influential_nodes_15_group.append(nodes_15[np.arange(len(nodes_15))!=i].tolist())
            influential_nodes_20_group.append(nodes_20[np.arange(len(nodes_20))!=i].tolist())
            normal_nodes_group.append(normal[np.arange(len(normal))!=i].tolist())
            bottom_nodes_group.append(bottom[np.arange(len(bottom))!=i].tolist())
            
    elif groupsize == int(N/20):
        influential_nodes_5_group = influential_nodes_5
        influential_nodes_10_group = influential_nodes_10
        influential_nodes_15_group = influential_nodes_15
        influential_nodes_20_group = influential_nodes_20
        normal_nodes_group = normal_nodes
        bottom_nodes_group = bottom_nodes
        
        S, t = simulate_spread_wrapper(G, influential_nodes_5_group, phi)
        influential_S_5.append(S)
        influential_t_5.append(t)
        S, t = simulate_spread_wrapper(G, influential_nodes_10_group, phi)
        influential_S_5.append(S)
        influential_t_5.append(t)
        S, t = simulate_spread_wrapper(G, influential_nodes_15_group, phi)
        influential_S_5.append(S)
        influential_t_5.append(t)
        S, t = simulate_spread_wrapper(G, influential_nodes_20_group, phi)
        influential_S_5.append(S)
        influential_t_5.append(t)
        S, t = simulate_spread_wrapper(G, normal_nodes_group, phi)
        influential_S_5.append(S)
        influential_t_5.append(t)
        S, t = simulate_spread_wrapper(G, bottom_nodes_group, phi)
        influential_S_5.append(S)
        influential_t_5.append(t)
        
        return [np.mean(influential_S_5), np.mean(influential_S_10), np.mean(influential_S_15), np.mean(influential_S_20), np.mean(influential_S_5 + influential_S_10), np.mean(influential_S_5 + influential_S_10 + influential_S_15), np.mean(influential_S_5 + influential_S_10 + influential_S_15 + influential_S_20), np.mean(normal_S), np.mean(bottom_S),
            np.mean(influential_t_5), np.mean(influential_t_10), np.mean(influential_t_15), np.mean(influential_t_20), np.mean(influential_t_5 + influential_t_10), np.mean(influential_t_5 + influential_t_10 + influential_t_15), np.mean(influential_t_5 + influential_t_10 + influential_t_15 + influential_t_20), np.mean(normal_t), np.mean(bottom_t)]
        
        
    ## for i in np.arange(len(influential_nodes_5)):
    ##     for j in np.arange(i, len(influential_nodes_5)):
    ##         influential_nodes_5_group.append([influential_nodes_5[i], influential_nodes_5[j]])
    ## for i in np.arange(len(influential_nodes_10)):
    ##     for j in np.arange(i, len(influential_nodes_10)):
    ##         influential_nodes_10_group.append([influential_nodes_10[i], influential_nodes_10[j]])            
    ## for i in np.arange(len(influential_nodes_15)):
    ##     for j in np.arange(i, len(influential_nodes_15)):
    ##         influential_nodes_15_group.append([influential_nodes_15[i], influential_nodes_15[j]])
    ## for i in np.arange(len(influential_nodes_20)):
    ##     for j in np.arange(i, len(influential_nodes_20)):
    ##         influential_nodes_20_group.append([influential_nodes_20[i], influential_nodes_20[j]])          
    ## for i in np.arange(len(normal_nodes)):
    ##     for j in np.arange(i, len(normal_nodes)):
    ##         normal_nodes_group.append([normal_nodes[i], normal_nodes[j]])
    ## for i in np.arange(len(bottom_nodes)):
    ##     for j in np.arange(i, len(bottom_nodes)):
    ##         bottom_nodes_group.append([bottom_nodes[i], bottom_nodes[j]])
    
    
    for node_list in influential_nodes_5_group:
        S, t = simulate_spread_wrapper(G, node_list, phi)
        influential_S_5.append(S)
        influential_t_5.append(t)    
    #for node_list in influential_nodes_10_groups:
    #    S, t = simulate_spread_wrapper(G, node_list, phi)
    #    influential_S_10.append(S)
    #    influential_t_10.append(t)
    for node_list in influential_nodes_10_group:
        S, t = simulate_spread_wrapper(G, node_list, phi)
        influential_S_10.append(S)
        influential_t_10.append(t)
    for node_list in influential_nodes_15_group:
        S, t = simulate_spread_wrapper(G, node_list, phi)
        influential_S_15.append(S)
        influential_t_15.append(t)
    for node_list in influential_nodes_20_group:
        S, t = simulate_spread_wrapper(G, node_list, phi)
        influential_S_20.append(S)
        influential_t_20.append(t)
    for node_list in normal_nodes_group:
        S, t = simulate_spread_wrapper(G, node_list, phi)
        normal_S.append(S)
        normal_t.append(t)
    for node_list in bottom_nodes_group:
        S, t = simulate_spread_wrapper(G, node_list, phi)
        bottom_S.append(S)
        bottom_t.append(t)
    
    return [np.mean(influential_S_5), np.mean(influential_S_10), np.mean(influential_S_15), np.mean(influential_S_20), np.mean(influential_S_5 + influential_S_10), np.mean(influential_S_5 + influential_S_10 + influential_S_15), np.mean(influential_S_5 + influential_S_10 + influential_S_15 + influential_S_20), np.mean(normal_S), np.mean(bottom_S),
            np.mean(influential_t_5), np.mean(influential_t_10), np.mean(influential_t_15), np.mean(influential_t_20), np.mean(influential_t_5 + influential_t_10), np.mean(influential_t_5 + influential_t_10 + influential_t_15), np.mean(influential_t_5 + influential_t_10 + influential_t_15 + influential_t_20), np.mean(normal_t), np.mean(bottom_t)]
    