import numpy
import ga

equation_inputs = [4,-2,3.5,5,-11,-4.7]
num_weights = 6
sol_per_pop = 8
num_parents_mating = 4

pop_size = (sol_per_pop,num_weights)
new_population = numpy.random.uniform(low=-5.12, high=5.12, size=pop_size)
print(new_population)

num_generations = 5
for generation in range(num_generations):
    print("Generation : ", generation)
    fitness = ga.cal_pop_fitness(equation_inputs, new_population)
    print("Fitness is:", fitness)

    parents = ga.select_mating_pool(new_population, fitness, 
                                      num_parents_mating)                               

    crossover = ga.crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_weights))
    print("Crossover is:", crossover) 

    mutation = ga.mutation(crossover)
    print("Mutation is:", mutation)

    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = mutation

    print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))

fitness = ga.cal_pop_fitness(equation_inputs, new_population)
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])
