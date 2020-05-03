import networkx as nx
import numpy as np


#########################################################################################
############################### Network Helper Functions  ###############################
#########################################################################################

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
        N = G.number_of_nodes()
        time_attrib = {i : value for i in range(N)}
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
        N = G.number_of_nodes()
        influence_attrib = { i : value for i in range(N) }
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
            if update_influence_directed(G_tmp, node, phi):
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

def run_simulation_RG_directed(N,p,phi=0.18,q=0.1):
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
    influential_nodes = degree_ordered_nodes[:int(q*N)]
#     normal_nodes = degree_ordered_nodes[int(q*N):]
    
    average = p * (N-1)
    lower, upper = int(np.floor(average)), int(np.ceil(average))
    normal_nodes = [x for x in G.nodes() if lower <= G.degree(x) <= upper ]

    influential_S = []
    influential_t = []
    normal_S = []
    normal_t = []
    
    ## Calculate the number of influenced nodes (S) and expected time of influenced nodes
    ## for each influential node
    for node in influential_nodes:
        S, t = simulate_spread_directed(G, node, phi)
        influential_S.append(S)
        influential_t.append(t)
        
    ## Calculate the number of influenced nodes (S) and expected time of influenced nodes
    ## for each normal node
    for node in normal_nodes:
        S, t = simulate_spread_directed(G, node, phi)
        normal_S.append(S)
        normal_t.append(t)

    return [np.mean(influential_S), np.mean(normal_S), np.mean(influential_t), np.mean(normal_t)]

def run_simulation_RG(N,p,phi=0.18,q=0.1):
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
    influential_nodes = degree_ordered_nodes[:int(q*N)]
#     normal_nodes = degree_ordered_nodes[int(q*N):]
    
    average = p * (N-1)
    lower, upper = int(np.floor(average)), int(np.ceil(average))
    normal_nodes = [x for x in G.nodes() if lower <= G.degree(x) <= upper ]

    influential_S = []
    influential_t = []
    normal_S = []
    normal_t = []
    
    ## Calculate the number of influenced nodes (S) and expected time of influenced nodes
    ## for each influential node
    for node in influential_nodes:
        S, t = simulate_spread(G, node, phi)
        influential_S.append(S)
        influential_t.append(t)
        
    ## Calculate the number of influenced nodes (S) and expected time of influenced nodes
    ## for each normal node
    for node in normal_nodes:
        S, t = simulate_spread(G, node, phi)
        normal_S.append(S)
        normal_t.append(t)

    return [np.mean(influential_S), np.mean(normal_S), np.mean(influential_t), np.mean(normal_t)]


def run_simulation_SF(N,n_avg,phi=0.18,q=0.1):
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
    influential_nodes = degree_ordered_nodes[:int(q*N)]
#     normal_nodes = degree_ordered_nodes[int(q*N):]
    
    ## Normal nodes are nodes with degree close to average
    lower, upper = int(np.floor(n_avg)), int(np.ceil(n_avg))
    normal_nodes = [x for x in G.nodes() if lower <= G.degree(x) <= upper ]

    influential_S = []
    influential_t = []
    normal_S = []
    normal_t = []
    
    ## Calculate the number of influenced nodes (S) and expected time of influenced nodes
    ## for each influential node
    for node in influential_nodes:
        S, t = simulate_spread(G, node, phi)
        influential_S.append(S)
        influential_t.append(t)
        
    ## Calculate the number of influenced nodes (S) and expected time of influenced nodes
    ## for each normal node
    for node in normal_nodes:
        S, t = simulate_spread(G, node, phi)
        normal_S.append(S)
        normal_t.append(t)

    return [np.mean(influential_S), np.mean(normal_S), np.mean(influential_t), np.mean(normal_t)]