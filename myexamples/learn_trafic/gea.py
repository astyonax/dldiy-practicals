#
# https://lethain.com/genetic-algorithms-cool-name-damn-simple/

from random import randint, random,uniform
from functools import partial
from operator import add,mul
import math
import lstm

#config begins here
def addsub(x,y):
    if random()>.5:
        return add(x,y)
    return add(x,-y)

def muldiv(x,y):
    if random()>.5:
        return mul(x,y)
    return mul(x,1./y)

def clip_gene(genes,key,x):
    'Clip the value of gene key within the boundary in the gene config.'
    m,M,_,_,dtype = genes[key]
    return dtype(min(max(m,x),M))

                    #   min,max,op,val
GENE_CFG = {'WINDOW' : [2,24,muldiv,1.5,int],
        'HIDDEN_DIM' : [1,32,muldiv,1.5,int],
        'WEIGHT_DECAY' : [-10,1,muldiv,1.5,int]
        }
EPOCHS = 200
PREFIX = 'GEA'
# config ends here
def individual():
    'Create a member of the population.'
    gene = {}
    for k,v in GENE_CFG.iteritems():
        m,M,_,_,dtype = v
        gene[k] = dtype(uniform(m,M))
    return gene


def population(count):
    """
    Create a number of individuals (i.e. a population).

    count: the number of individuals in the population
    length: the number of values per individual
    min: the minimum possible value in an individual's list of values
    max: the maximum possible value in an individual's list of values

    """
    return [ individual() for x in xrange(count) ]


def fitness(individual):
    """
    Determine the fitness of an individual. Higher is better.

    individual: the individual to evaluate
    target: the target number individuals are aiming for
    """

    # lstm.main is already momoized once every EV epoch EV is paramter of main
    # default: EV=1000
    return lstm.main(EPOCHS=EPOCHS,prefix=PREFIX,**individual)

def grade(pop):
    'Find average fitness for a population.'
    summed = sum(fitness(x) for x in pop)
    return summed / (len(pop) * 1.0)

def evolve(pop, retain=0.5, random_select=0.05, mutate=0.9):
    _graded = [ (fitness(x), x) for x in pop]
    print('fitness',sorted(_graded)[0][0])
    graded = [ x[1] for x in sorted(_graded)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]
    # randomly add other individuals to
    # promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    # mutate some individuals
    for idx,individual in enumerate(parents):
        if mutate > random() and idx>=retain_length:
            key_to_mutate = individual.keys()[randint(0, len(individual)-1)]
            op = GENE_CFG[key_to_mutate][2]
            x  = individual[key_to_mutate]
            y  = GENE_CFG[key_to_mutate][3]
            individual[key_to_mutate] = clip_gene(GENE_CFG,key_to_mutate,op(x,y))

    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    assert(parents_length!=1)
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        random_gene = GENE_CFG.keys()[randint(0,2 )]
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) / 2
            child = male.copy()
            child[random_gene]= female[random_gene]
            children.append(child)
    parents.extend(children)
    return parents,_graded


if __name__ == '__main__':
    pop = population(50)
    for j in xrange(20):
        pop, graded = evolve(pop,)
    import pickle
    with open('GEA_optimal.pkl','wb') as fout:
        pickle.dump(graded,fout)
