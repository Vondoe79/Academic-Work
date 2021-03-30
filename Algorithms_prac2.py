# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:48:10 2019

@author: hadj0823
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

plt.style.use('ggplot')

## inputs of ---> a Simple Random Search Algorithm
max_iter = 1000
prob_size = 2
search_space = [-5, 5]


def init_solu(prob_size, search_space):
    return [min(search_space) + np.random.randint(max(search_space) - min(search_space)) \
            for i in range(prob_size)]


def obj_func(x1, x2):
    return x1 ** 2 + x2 ** 2


best = []
candidate = []
candidate_sorted = []
CosT = []

for iteR in range(max_iter):
    candidate.append([])
    candidate[iteR].append(init_solu(prob_size, search_space))
    cost = obj_func(candidate[iteR][0][0], candidate[iteR][0][1])
    CosT.append(cost)
    candidate[iteR].append(cost)
    candidate_sorted.append(sorted(candidate, key=lambda X: X[1], reverse=False))
    best.append(candidate_sorted[iteR][0])

    if candidate[iteR][1] < best[-1][1]:
        best = candidate_sorted[-1]

#    print ("Generation {}'s sorted Candidates are:--> {}".format(iteR, candidate_sorted))
print("The best solution found is:--> {best[-1]}")

# plot the cost function values
fig, ax = plt.subplots(figsize=(15, 6));
for i in range(len(CosT)):
    plt.plot(sorted(CosT, reverse=True))
    plt.title("Solution Plot of Random Search Heuristic Algorithm")
    plt.xlabel("Nbr Generations/Iterations")
    plt.ylabel("Obj. Function Value(Cost)")
plt.show()
