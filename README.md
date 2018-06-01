# TREE
Optimization on Tree Ensembles and Dominating Point.

This project is written in Python, and use Gurobi as optimization solver.

This project takes tree ensemble as input, and use Benders Decomposition method to solve large scale optimization problem.

By solving MIP, and using bisection, we can get the solution, which is then called dominating point.

prepare.py is written to process the tree ensemble as input, including how to find the leaf and its prediction, left-child tree and so on.

MIP_solver.py and MIP_solver_2.py are written to solve the optimization problem calling Gurobi in Python, and use bisection to get dominating point.

Examples include 2-dimension case, 3-dimension case, high-dimension case and a case on real dataset, the concrete.csv.

In order to get real dataset, we import AppliedPredictiveModeling package in R and save it as csv file.

All our work is to help with the preparation of work in importance sampling prediction.pdf

The Algorithm of optimization on tree ensemble is based on the paper "Optimization of Tree Ensemble", please refer to the pdf file.
