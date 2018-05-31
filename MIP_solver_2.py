import pandas as pd
import numpy as np
from prepare import *
from gurobipy import *

def MIP_solver_2(trees_given, flag_given, eta_given,x_orig):     
    eta=eta_given
    trees=trees_given
    flag=flag_given
    lama=np.zeros(len(trees))
    for i in range(len(trees)):
        lama[i] = 1 / len(trees) 
        
    #create initial solution for alpha, beta, gamma
    alpha={}
    beta={}
    gamma={}
    for i in range(len(trees)):
        for j in splits(trees,i):
            #create variables
            alpha[i,j]=0
            beta[i,j]=0
    for i in range(len(trees)):
        gamma[i]=0
        for j in leaves(trees,i):
            gamma[i] = max(gamma[i],prediction(trees,i,j,flag))    

    def add_constraint_2(model, where):
        if where == GRB.Callback.MIPSOL:
            sol_theta=model.cbGetSolution([model._vars_theta[i] for i in range(len(model._vars_theta))])
            sol_X_one={}
            for i in total_split_variable(trees):
                for j in range(K(trees,i)):                
                    sol_X_one[i,j]=model.cbGetSolution(model._vars_X_one[i,j])
            alpha_new={}
            beta_new={}
            gamma_new={}
            for i in range(len(trees)):
                l_optimal=GETLEAF(trees,i,sol_X_one)
                for j in splits(trees,i):
                    temp1=0
                    if j in as_right_leaf(trees,i,l_optimal):
                        for l in left_leaf(trees,i,j):
                            temp1=max(temp1,(prediction(trees,i,l,flag)-prediction(trees,i,l_optimal,flag)))
                    alpha_new[i,j]=temp1
                    temp2=0
                    if j in as_left_leaf(trees,i,l_optimal):
                        for l in right_leaf(trees,i,j):
                            temp2=max(temp2,prediction(trees,i,l,flag)-prediction(trees,i,l_optimal,flag))
                    beta_new[i,j]=temp2
                gamma_new[i]=prediction(trees,i,l_optimal,flag)
            for i in reversed(range(len(trees))):
                expr=0
                for s in splits(trees, i):
                    expr=expr+alpha_new[i,s]*sol_X_one[V(trees,i,s),C(trees,i,s)]+beta_new[i,s]*(1-sol_X_one[V(trees,i,s),C(trees,i,s)])
                expr=expr+gamma_new[i]-sol_theta[i]
                if expr < 0:   
                    print("Find a violated constraint!")
                    model.cbLazy(quicksum(alpha_new[i,s]*model._vars_X_one[V(trees,i,s),C(trees,i,s)] for s in splits(trees,i)) + quicksum(beta_new[i,s]*(1-model._vars_X_one[V(trees,i,s),C(trees,i,s)]) for s in splits(trees,i)) + gamma_new[i] - model._vars_theta[i] >= 0)                
                    

    #create a new model
    m = Model("tree_ensemble")

    #create variables
    X={}
    theta={}
    X_one={}

    for i in total_split_variable(trees):
        X[i]=m.addVar(name='X'+str(i))
        for j in range(K(trees,i)):
            X_one[i,j]=m.addVar(vtype=GRB.BINARY, name='X_one'+str(i)+'_'+str(j))
    for i in range(len(trees)):
        theta[i]=m.addVar(lb=-GRB.INFINITY, name='theta' + str(i))
    m.update()

    # Set objective
    m.setObjective(quicksum(lama[i]*theta[i] for i in range(len(trees))),GRB.MAXIMIZE)
    m.update()

    # Add constraint
    for i in range(len(trees)):
        m.addConstr(quicksum(alpha[i,s]*X_one[V(trees,i,s),C(trees,i,s)] for s in splits(trees,i)) + quicksum(beta[i,s]*(1-X_one[V(trees,i,s),C(trees,i,s)]) for s in splits(trees,i)) + gamma[i] - theta[i] >= 0) 

    for i in range(len(trees)):
        for j in splits(trees,i):
            m.addConstr((X_one[V(trees,i,j),C(trees,i,j)] == 1) >> (X[V(trees,i,j)] - split_values(trees,V(trees,i,j))[C(trees,i,j)] <= 0) )
            m.addConstr((X_one[V(trees,i,j),C(trees,i,j)] == 0) >> (X[V(trees,i,j)] - split_values(trees,V(trees,i,j))[C(trees,i,j)] >= 1e-5) )

    for i in total_split_variable(trees):
        for j in range(K(trees,i)-1):
            m.addConstr(X_one[i,j] - X_one[i,j+1] <= 0)
    
    
    
    m.addConstr(quicksum(X[i]*x_orig[i] - x_orig[i]*x_orig[i] for i in total_split_variable(trees)) <= -1e-1)
    m.addConstr(X[0]*X[0]+X[1]*X[1] <= eta)
    m.update()

    m._vars_X_one=X_one
    m._vars_theta=theta
    m.params.LazyConstraints = 1
    m.optimize(add_constraint_2)
    
    optimal_value=0
    for i in range(len(trees)):
        optimal_value=optimal_value+lama[i]*theta[i].x    

    optimal_solution=np.zeros(len(X))
    for i in range(len(X)):
        optimal_solution[i]=X[i].x
    
    return [optimal_value, optimal_solution]



def bisection_2(trees,flag,eta_1, eta_2,baseline,x_orig):
    if eta_1 >= eta_2:
        print("Oh-oh, eta_1 should be lower than eta_2~\n")
    else:
        iteration=0
        sol_1={}
        x_1={}
        sol_2={}
        x_2={}
        sol_new={}
        x_new={}
        eta_new=(eta_1+eta_2)/2
        [sol_1[iteration],x_1[iteration]]=MIP_solver_2(trees,flag,eta_1,x_orig)
        print(x_1[iteration],"\n")
        [sol_2[iteration],x_2[iteration]]=MIP_solver_2(trees,flag,eta_2,x_orig)
        print(x_2[iteration],"\n")
        [sol_new[iteration],x_new[iteration]]=MIP_solver_2(trees,flag,eta_new,x_orig)
        if sol_1[iteration]<baseline and sol_2[iteration]>baseline:
            while abs(sol_new[iteration]-baseline)>1e-2 and abs(min(abs(eta_new-eta_1),abs(eta_new-eta_2)))>1e-2:
                print("\n",eta_new,"\n")
                print("\n",x_new[iteration],"\n")
                if sol_new[iteration]<baseline:
                    iteration=iteration+1
                    eta_1=eta_new
                    eta_new=(eta_new+eta_2)/2                    
                    [sol_1[iteration],x_1[iteration]]=[sol_new[iteration-1],x_new[iteration-1]]
                    [sol_new[iteration],x_new[iteration]]=MIP_solver_2(trees,flag,eta_new,x_orig)
                else:
                    iteration=iteration+1
                    eta_2=eta_new
                    eta_new=(eta_1+eta_new)/2
                    [sol_2[iteration],x_2[iteration]]=[sol_new[iteration-1],x_new[iteration-1]]
                    [sol_new[iteration],x_new[iteration]]=MIP_solver_2(trees,flag,eta_new,x_orig)
        elif sol_1[iteration]>baseline and sol_2[iteration]<baseline:
            while abs(sol_new[iteration]-baseline)>1e-2 and abs(min(abs(eta_new-eta_1),abs(eta_new-eta_2)))>1e-2:
                if sol_new[iteration]<baseline:
                    iteration=iteration+1
                    eta_2=eta_new
                    eta_new=(eta_new+eta_1)/2                    
                    [sol_2[iteration],x_2[iteration]]=[sol_new[iteration-1],x_new[iteration-1]]
                    [sol_new[iteration],x_new[iteration]]=MIP_solver_2(trees,flag,eta_new,x_orig)
                else:
                    iteration=iteration+1
                    eta_1=eta_new
                    eta_new=(eta_2+eta_new)/2
                    [sol_1[iteration],x_1[iteration]]=[sol_new[iteration-1],x_new[iteration-1]]
                    [sol_new[iteration],x_new[iteration]]=MIP_solver_2(trees,flag,eta_new,x_orig)     

        else:
            print("The initial value of eta_1 and eta_2 is not correct! Make eta_1 smaller or eta_2 larger!")
        for i in reversed(range(iteration)):
            if sol_new[i]<baseline:
                continue
            else:
                return [sol_new[i],x_new[i]]
            

