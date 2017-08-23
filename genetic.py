import random, array, numpy
from deap import base, creator, tools, algorithms
import network

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randrange, 10, 41, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalNetwork(individual):
	result = network.execute(individual[0]+60, individual[1]+30, individual[2]+10, individual[3]/10.0, individual[4]/100.0, "genetic_result.txt")
	return (result,)

toolbox.register("evaluate", evalNetwork)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
	random.seed(64)

	pop = toolbox.population(n=20)
	hof = tools.HallOfFame(5)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", numpy.mean)
	stats.register("std", numpy.std)
	stats.register("min", numpy.min)
	stats.register("max", numpy.max)

	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=True)

	print log
	print hof
	return pop, log, hof

if __name__ == "__main__":
	main()
