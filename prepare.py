import pandas as pd
import numpy as np
#from sklearn.externals.six import StringIO  
#from IPython.display import Image
#from sklearn.tree import export_graphviz
#import pydotplus
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

def get_rf(X,Y,n,flag): #given flag, observation and number of the tree you want to have, get the random forest 
    if flag==1:         
        rf = RandomForestRegressor(n_estimators=n,random_state=1)
        rf = rf.fit(X,Y)
    elif flag==0:
        rf = RandomForestClassifier(n_estimators=n,random_state=1)
        rf = rf.fit(X,Y)        
    else:
        print("flag can only take values 0 and 1")
    return rf

def get_input(rf): #this function captures the input tree ensemble and return list of trees embedded
    trees=list()
    for i in range(rf.n_estimators): 
        trees.append(rf.estimators_[i])
    return trees

def is_it_leaf(trees_given,t): #this function returns an array of the boolean value, telling if it is leaf of the tree t
    tree=trees_given[t].tree_
    n_nodes=tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [0]  # seed is the root node id
    while len(stack) > 0:
        node_id = stack.pop()
        if (children_left[node_id] != children_right[node_id]):
            stack.append(children_left[node_id])
            stack.append(children_right[node_id])
        else:
            is_leaves[node_id] = True
    return is_leaves

def leaves(trees_given,t):
    return np.arange(trees_given[t].tree_.node_count)[is_it_leaf(trees_given,t)==True]
      
def GETLEAF(trees_given,t,x_given): #this function determines the leaf to which tree t maps x
    v=0
    while is_it_leaf(trees_given,t)[v]==False:
        if x_given[V(trees_given,t,v),C(trees_given,t,v)]==1:
            v=trees_given[t].tree_.children_left[v]
        else:
            v=trees_given[t].tree_.children_right[v]
    return v

def total_split_variable(trees_given): #this function returns the set of independent variables that are used in split conditions
    feature_set=set([])
    for tree in trees_given:
        feature=tree.tree_.feature[tree.tree_.feature>=0]
        feature_set=feature_set|set(feature)
    return feature_set

def V(trees_given,t,s): #this function returns variable that participates in split s
    tree=trees_given[t].tree_
    feature=tree.feature[s]
    return feature

def split_values(trees_given,i): #this function returns array of unique split points in ascendng order
    values=np.array([])
    if i in total_split_variable(trees_given): 
        for tree in trees_given:
            feature=set(tree.tree_.feature[tree.tree_.feature>=0])
            if i in feature:
                values=np.append(values,tree.tree_.threshold[tree.tree_.feature==i])
    values=np.unique(np.sort(values))
    return values

def C(trees_given,t,s): #set of values of variables i that participate in split
    #the expression in the paper is not right, since there is only one threshold in each split
    threshold=trees_given[t].tree_.threshold[s]
    feature=V(trees_given,t,s)
    return int(np.where(split_values(trees_given,feature)==threshold)[0])

def K(trees_given,i): #number of unique split points
    return split_values(trees_given,i).shape[0]

def prediction(trees_given,t,l,flag): #prediction of tree t, leaf l
    tree=trees_given[t].tree_
    if flag==0: #this is a classification tree and return which class the leaf predicts
        return np.argmax(tree.value[l,0,:])
    else: 
        return tree.value[:,0,0][l] #this is a regression tree 
'''
def y(trees_given,t,x_given):  #y_t,l=1 if l = l*; otherwise, 0
    l_star=GETLEAF(trees_given,t,x_given)
    tree=trees_given[t].tree_
    node=tree.node_count
    y_return=np.zeros(node)
    y_return[l_star]=1
    return y_return
'''
def splits(trees_given,t): #return an array of splits(not leaf) of tree
    return np.arange(trees_given[t].tree_.node_count)[is_it_leaf(trees_given,t)==False]

def right_leaf(trees_given,t,s):  #return a list of all the right leaf of tree t, node s
    right_leaves=[]
    tree=trees_given[t].tree_
    n_nodes=tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    stack = [s]
    node_id = stack.pop()
    if (children_left[node_id] != children_right[node_id]):
        stack.append(children_right[node_id])
    else:
        return right_leaves
    while len(stack) > 0:
        node_id = stack.pop()
        if (children_left[node_id] != children_right[node_id]):
            stack.append(children_left[node_id])
            stack.append(children_right[node_id])
        else:
            right_leaves.append(node_id)
    return right_leaves

def as_right_leaf(trees_given,t,l): #return an array of splits whose right leaf is l
    split=np.array([])
    tree=trees_given[t].tree_
    n_nodes=tree.node_count
    for i in range(n_nodes):
        if l in set(right_leaf(trees_given,t,i)):
            split=np.append(split,np.array([i])) 
    return split

def left_leaf(trees_given,t,s): #return a list of all the left leaf of tree t, node s
    left_leaves=[]
    tree=trees_given[t].tree_
    n_nodes=tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    stack = [s]
    node_id = stack.pop()
    if (children_left[node_id] != children_right[node_id]):
        stack.append(children_left[node_id])
    else:
        return left_leaves
    while len(stack) > 0:
        node_id = stack.pop()
        if (children_left[node_id] != children_right[node_id]):
            stack.append(children_left[node_id])
            stack.append(children_right[node_id])
        else:
            left_leaves.append(node_id)
    return left_leaves

def as_left_leaf(trees_given,t,l): #return an array of splits whose left leaf is l
    split=np.array([])
    tree=trees_given[t].tree_
    n_nodes=tree.node_count
    for i in range(n_nodes):
        if l in set(left_leaf(trees_given,t,i)):
            split=np.append(split,np.array([i])) 
    return split

def x_to_one(trees_given,x_given): #given an observation of X, x_V(s),j = 1 if j belongs to C(s)
    x_one={}
    for i in range(len(trees_given)):
        for j in splits(trees_given,i):
            if x_given[V(trees_given,i,j)] - split_values(trees_given,V(trees_given,i,j))[C(trees_given,i,j)] <= 0:
                x_one[V(trees_given,i,j),C(trees_given,i,j)] = 1
            else:
                x_one[V(trees_given,i,j),C(trees_given,i,j)] = 0
    return x_one


def plot_decision_tree(trees_given, i): #plot the decision path for tree i, given trees
    dot_data = StringIO()
    export_graphviz(trees_given[i], out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph.create_png())

def create_baseline(Y,perc):
    baseline=np.array(sorted(Y,reverse=True))[int(perc*len(Y))]
    return baseline