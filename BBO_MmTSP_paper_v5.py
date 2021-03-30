# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:29:59 2017
@author: Hayford Adjavor
"""
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from itertools import tee
from datetime import datetime
import profile
import sys

PopulationSize = 5
prob_mutation = 0.01
mu = np.zeros(PopulationSize)
Lambda = np.zeros(PopulationSize)
E = 1
I = 1
max_gen = 100
Num_affected_areas = 14
Num_Depots = 1
Num_salesmen = 1
ProblemDimension = Num_affected_areas + 2 * Num_salesmen

depot_list = range(Num_Depots)
affected_list = range(Num_Depots, Num_affected_areas + Num_Depots)
fit_val = np.zeros((PopulationSize), dtype=np.int64)
Mean = np.zeros((max_gen), dtype=float)
Mean_2 = np.zeros((max_gen), dtype=float)
best_in_gen = np.zeros((max_gen), dtype=float)
best_fit = np.zeros((max_gen), dtype=np.int64)
Optimal_Route = np.zeros((ProblemDimension), dtype=np.int64)
Population = np.zeros((PopulationSize, ProblemDimension + Num_salesmen + 1), dtype=np.int64)
total = ProblemDimension + Num_salesmen

SelectInd = []

random.seed(10)

start_time = datetime.now()


##Inition population: -----------------------
def solution_ecoding():
    global Population_list

    Population_list = []
    for p in range(PopulationSize):
        solutionPermutation = np.random.permutation(affected_list).tolist()
        assigned_areas = []
        num_areas = []
        solution = [[], []]
        for i in range(Num_salesmen):
            if i != Num_salesmen - 1:
                if (len(solutionPermutation) / float((Num_salesmen - i + 1))) != 1:
                    splt_point = np.random.randint(1, len(solutionPermutation) - (Num_salesmen - i - 1))
                else:
                    splt_point = 1
                assigned_areas.append(solutionPermutation[:splt_point])
                del solutionPermutation[:splt_point]
            else:
                assigned_areas.append(solutionPermutation)
                del solutionPermutation
                ####--stop here in the reading:
            if len(assigned_areas[i]) == 1:
                s_depot = int(np.random.choice(depot_list))
                e_depot = int(np.random.choice(depot_list))
                while s_depot == e_depot:
                    s_depot = int(np.random.choice(depot_list))
                    e_depot = int(np.random.choice(depot_list))
            else:
                s_depot = int(np.random.choice(depot_list))
                e_depot = int(np.random.choice(depot_list))
            assigned_areas[i].insert(0, s_depot)
            assigned_areas[i].append(e_depot)
            num_areas.append(len(assigned_areas[i]))
            solution[0] = solution[0] + assigned_areas[i]
        solution[1] = solution[1] + num_areas

        Population_list.append(solution)

    # ----Below lines of codes computes the Obj. Function
    for p in Population_list:
        dis = 0
        s_n = 0
        for s in range(Num_salesmen):
            for n in range(p[1][s] - 1):
                dis += distance_list[p[0][s_n]][p[0][s_n + 1]]
                s_n += 1
            s_n += 1

        p.append(dis)

    return Population_list


###_______________________________________________________________________#
##_Data files:__ testMatrix_15.csv __ testMatrix_17_v1.csv ---#
##__ testMatrix_26_v1.csv __ testMatrix_29_v1.csv __ testMatrix_42_v1.csv
##__ katriMatrix_2.csv __ testMatrix_416_v1 __ testMatrix_416_v1_30_instance.csv
def readcsvFile():
    data = pd.read_csv('testMatrix_15.csv', index_col=None, skiprows=None)
    return data


distance = readcsvFile()
distance_list = np.array(distance, dtype=float)


###______________________________________________________#
###_create an initial population
def init_population():
    pop_list = solution_ecoding()
    for h in range(PopulationSize):
        for r in range(ProblemDimension):
            Population[h, r] = pop_list[h][0][r]
        for k in range(Num_salesmen):
            Population[h, r + k + 1] = pop_list[h][1][k]
        Population[h, r + k + 2] = pop_list[h][2]

    return Population


###_the fitness function:
def fitness(h):
    global total
    s_n = 0
    distance = 0
    for s in range(ProblemDimension, total):
        num_visit = h[s]
        for x in range(s_n, s_n + num_visit - 1):
            distance += (distance_list[h[x]][h[x + 1]])
        s_n += num_visit

    return distance


###_Sort population based on fitness:
def sort(Pop):
    for i in range(PopulationSize):
        fit_val[i] = fitness(Pop[i, :])
    indices = np.argsort(fit_val)  # sorting by indices
    Pop = Pop[indices]  # apply sorted indices to the population

    return Pop


###_compute mu and lambda rates:
def lamda_Mu(PopulationSize):
    for i in range(PopulationSize):
        mu[i] = E - (i) / float((PopulationSize + 1))  # emigration rate
        Lambda[i] = I - mu[i]  # immigration rate

    return Lambda, mu


###_the main BBO Algorithm:
def main_BBO(pop):
    global Lambda, Population, mu, Island1, Island2, max_gen, best_fit, Mean, \
        fitness_cmb, total, combind, best_in_gen, Mean_2, Optimal_Route, SelectInd

    Lambda, mu = lamda_Mu(PopulationSize)
    for g in range(max_gen):
        # Perform migration operator:
        delta = np.random.randint(Num_affected_areas + Num_Depots)
        for k in range(PopulationSize):
            # select Hi based on mu
            if np.random.uniform(0, 1) < Lambda[k]:
                # begin roulette wheel
                RandomNum = np.random.uniform(0, 1) * sum(mu)
                Select = mu[0]
                SelectIndex = 0  # index for selected solution in population
                while (RandomNum > Select) and (SelectIndex < (PopulationSize - 1) / 3):
                    SelectIndex += 1
                    Select += mu[SelectIndex]
                SelectInd.append(SelectIndex)  # printing
                Island1 = pop[SelectIndex, :ProblemDimension].tolist()
                Island2 = pop[SelectIndex + 1, :ProblemDimension].tolist()
                if delta in Island1:
                    index1 = Island1.index(delta)
                    if Island2[index1] != delta:
                        if delta in range(Num_Depots, Num_affected_areas + Num_Depots) and \
                                Island2[index1] in range(Num_Depots, Num_affected_areas + Num_Depots):
                            index2 = Island2.index(delta)
                            Island2[index2] = Island2[index1]
                            Island2[index1] = delta
                            pop[k, :ProblemDimension] = Island2
                            fit = fitness(pop[k, :])
                            pop[k, total] = fit

        pop = sort(pop)
        ###__Perform mutation - 2-opts - Swap Operator:
        for m in range(PopulationSize):
            if np.random.uniform(0, 1) <= prob_mutation:
                RN1 = np.random.randint(ProblemDimension)
                RN2 = np.random.randint(ProblemDimension)
                while RN1 == RN2:
                    RN2 = np.random.randint(
                        ProblemDimension)  # this could as well be RN1; just so the two are not the same
                RN1_Dep = pop[m][RN1]
                RN2_Dep = pop[m][RN2]
                if pop[m][RN1] in range(Num_Depots) and pop[m][RN2] in range(Num_Depots):
                    pop[m][RN1] = RN2_Dep
                    pop[m][RN2] = RN1_Dep
                elif pop[m][RN1] in range(Num_Depots, Num_affected_areas + Num_Depots) and \
                        pop[m][RN2] in range(Num_Depots, Num_affected_areas + Num_Depots):
                    pop[m][RN1] = RN2_Dep
                    pop[m][RN2] = RN1_Dep
                pop = sort(pop)
        Lambda, mu = lamda_Mu(PopulationSize)
        best_fit[g] = fitness(pop[0, :])

        combined = np.concatenate((Population, pop))
        #        combined = np.unique(combined, axis=0) #removing duplicates
        fitness_cmb = np.zeros((len(combined)))  # holds the fitness of the combined parent and children
        for i in range(len(combined)):
            fitness_cmb[i] = fitness(combined[i, :])  # store the fitness of the combined parent and children
        indices = np.argsort(fitness_cmb)  # sorted index based on the fitness value (ascending)
        combined = combined[indices]  # holds the sorted fitness based on the index
        Population = combined[:PopulationSize, :]  # slice the top from combined and replace the parents
        for n in range(PopulationSize):
            fit_val[n] = fitness(Population[n, :])
        Mean[g] = np.mean(fit_val)
        Mean_2[g] = np.mean(fitness_cmb)

    print("Best_Tour: \n", list(Population[0, :]))
    Optimal_Route = Population[0, :][:-3]
    print("Optimum Route is: \n", Optimal_Route)


if __name__ == "__main__":
    Population = init_population()
    Population = sort(Population)
    main_BBO(Population)
    finished_time = datetime.now()
    Elapsed_time = finished_time - start_time
    print('Time elapsed - hh:mm:ss.ms - is: {}'.format(Elapsed_time))

#######________________________________________####
###--Ploting the graph of convergence-----:
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
line1, = ax1.plot(Mean, 'b-', label='Single')
line2, = ax2.plot(Mean_2, 'r-', label='Combined')
plt.title('BBO Avg. & Combined Convergence Plot', loc='center', fontsize='large')
ax1.set_ylabel('Average Single Fitness Value', color='blue')
ax2.set_ylabel('Avg. Combined Fitness Value', color='red')
ax1.set_xlabel('No.Generations (iterations)')
plt.legend(handles=[line1, line2])
plt.grid(True)
plt.savefig('BBO_MmTSP_SolConvergence_Plot.png', dpi=100)
plt.show()
###------------------------------------------


####_Plotting optimal TUPLE path list and plotting with networkx:_######
# def pairwise(iterable):
#    a, b = tee(iterable)
#    next(b, None)
#    return list(zip(a, b))
# Optimal_Route_BBO = pairwise(Optimal_Route)
# redundant_list = [(1,1),(1,0),(0,1),(0,0)]
# for tup in Optimal_Route_BBO:
#    if tup in redundant_list:
#        Optimal_Route_BBO.remove(tup)
#    else:
#        continue
# print "Route=>>", Optimal_Route_BBO
#
#########################################
####___Plotting with networkx: ___#
# G = nx.MultiDiGraph()
# G.add_edges_from(Optimal_Route_BBO)
#
# nx.draw_networkx(G, with_labels=True, arrowsize=20, node_color=\
#                 ['red' if 0<=i<=1  else 'orange' for i in G.nodes()], \
#                 node_size=[G.degree(i)*200 for i in G.nodes()], edge_color='red')
# plt.title('BBO Optimal Route/Path of Salesmen', fontsize= 'x-large', loc='center')
# plt.savefig('BBO_Optimal_Tour_Networkx.png', dpi=100, bbox_inches='tight')
# plt.show()
