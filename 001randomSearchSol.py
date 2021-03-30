# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:51:28 2017

@author: Vondoe79
"""

import random

def cost(solution):
    sum = 0
    for x in solution:
        sum += x*x
    return sum

def randomSolution(searchSpace):
    newSolution = []
    for bounds in searchSpace:
        (lowerBound, upperBound) = bounds
        newSolution.append(lowerBound + random.random()*(upperBound - lowerBound))
    return newSolution

def randomSearch(numIterations, searchSpace):
    best = None
    bestCost = 100000000000
    
    for i in range(numIterations):
        candidateSolution = randomSolution(searchSpace)
        candidateCost = cost(candidateSolution)
        if (candidateCost < bestCost):
            best = candidateSolution
            bestCost = candidateCost
            
    return best

problem_size = 2
search_space = [(-5,5),(-5,5)]
max_iter = 100
best = randomSearch(max_iter, search_space)
print (f'Best solution ==>>  {best}')