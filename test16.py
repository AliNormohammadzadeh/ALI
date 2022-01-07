from typing import  List , Optional,  Callable, Tuple
from random import choices , uniform , randint , randrange , random
import matplotlib.pyplot as plt

genome = list[int]
Population = list[genome]
CrossoverFunc =Callable[[genome,genome],Tuple[genome,genome]]
fitnessFunc = Callable[[genome],int]
MutationFunc = Callable[[genome], genome]
fitnessFuncChoice = Callable[[genome],int]
SelectionFunc = Callable[[Population, fitnessFuncChoice], Tuple[genome,genome]]
PopulationFunc = Callable[[],Population]
fitness_List = []


def generate_genome(length: int = 4)->genome:
    return choices([0,1],k= length)



def Population(size : int = 6 ) -> Population:
    return [generate_genome() for _ in range(size)]

def singele_point_crossover(a :genome , b : genome )->Tuple[genome , genome]:
    cross_rate = uniform(0,1)
    length = len(a)
    if cross_rate < 0.3 : 
        return None , None
    else :
        p = randint(1, length-1)
        return a[0:p] + b[p:] , b[0:p] + a[p:]
    
def mutation(genome: genome, num: int = 1, probability: float = 0.01) -> genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome    

def fitness_func(genome)->int:
    sum = 0 
    goal = 0
    for i in range(len(genome)):
        sum = sum + genome[i] * pow(2,len(genome)-i-1)
    goal = -24 * sum + 4* pow(sum,2)
    return -(goal)

def fitnessFuncChoice(genome)->int:
    sum = 0 
    goal = 0
    min = 540
    for i in range(len(genome)):
        sum = sum + genome[i] * pow(2,len(genome)-i-1)       
    goal = -24 * sum + 4* pow(sum,2)
    return -(goal) + min +1

def selection_pair(population:Population, fitness_func : fitnessFuncChoice,k :int = 6)->Population:
    return choices(
        population = population,
        weights=[fitness_func(gene) for gene in population],
        k = 6
    )
 
def sort_population(population:Population , fitness_func :fitnessFunc)->Population:
    return sorted(population,key=fitness_func,reverse=True)

def population_fitness(population: Population, fitness_func: fitnessFunc) -> int:
    return sum([fitness_func(genome) for genome in population])

def show(fitness_list):
    x = []
    y = fitness_list
    for i in range(len(y)):
        x.append(i)
    fig, ax = plt.subplots()  
    ax.plot(x, y); 
    return ax  

generate_limit = 100
pop = Population(6)

for j in range(generate_limit):
    parents = selection_pair(pop,fitnessFuncChoice)
    offspring = []
    i = 0
    for i in range(3):
        sp = 2*i
        spring_a , spring_b = singele_point_crossover(parents[sp],parents[sp+1])
        if spring_a != None and spring_b != None:
            spring_a = mutation(spring_a)
            spring_b = mutation(spring_b)
        if spring_a != None and spring_b != None:
            pop += [spring_a,spring_b]

    pop=selection_pair(pop,fitnessFuncChoice)

    fitness_List.append(population_fitness(pop,fitnessFuncChoice)/3456) 
    if fitness_List[j] >= 1 :
        break

print(fitness_List)            
   
