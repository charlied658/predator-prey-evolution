import random
import copy
import json

from godot_interface import evaluate_batch


# Configuration
POPULATION_SIZE = 30
GENERATIONS = 30
GENOME_LENGTH = 66

MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.2
ELITE_COUNT = 2
TOURNAMENT_SIZE = 3


def random_genome():
    return [random.uniform(-1, 1) for _ in range(GENOME_LENGTH)]


def mutate(genome):
    child = genome.copy()

    for i in range(len(child)):
        if random.random() < MUTATION_RATE:
            child[i] += random.gauss(0, MUTATION_STRENGTH)

    return child


def crossover(parent1, parent2):
    return [
        parent1[i] if random.random() < 0.5 else parent2[i]
        for i in range(GENOME_LENGTH)
    ]


def tournament_selection(population, fitnesses):
    candidates = random.sample(
        range(len(population)),
        TOURNAMENT_SIZE
    )

    winner = max(candidates, key=lambda i: fitnesses[i])

    return population[winner]


def evaluate_population(population, prey_genome):
    encounters = [
        {
            "predator_genome": predator,
            "prey_genome": prey_genome
        }
        for predator in population
    ]

    results = evaluate_batch(encounters)

    return [result["predator_fitness"] for result in results]


def evolve():
    population = [
        random_genome()
        for _ in range(POPULATION_SIZE)
    ]

    # Fixed prey opponent
    prey_genome = random_genome()

    best_fitness_history = []

    overall_best_fitness = float("-inf")
    overall_best_genome = None

    for generation in range(GENERATIONS):
        fitnesses = evaluate_population(population, prey_genome)

        best_index = max(
            range(POPULATION_SIZE),
            key=lambda i: fitnesses[i]
        )

        best_fitness = fitnesses[best_index]

        if best_fitness > overall_best_fitness:
            overall_best_fitness = best_fitness
            overall_best_genome = population[best_index].copy()

        average_fitness = sum(fitnesses) / POPULATION_SIZE

        best_fitness_history.append(best_fitness)

        print(
            f"Generation {generation + 1}/{GENERATIONS} "
            f"| Best: {best_fitness:.2f} "
            f"| Average: {average_fitness:.2f}"
        )

        # Elitism
        ranked = sorted(
            range(POPULATION_SIZE),
            key=lambda i: fitnesses[i],
            reverse=True
        )

        next_population = [
            copy.deepcopy(population[i])
            for i in ranked[:ELITE_COUNT]
        ]

        # Reproduction
        while len(next_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            child = crossover(parent1, parent2)
            child = mutate(child)

            next_population.append(child)

        population = next_population

    # Save the best genome from the final evaluated generation
    with open("best_predator.json", "w") as f:
        json.dump(
            {
                "genome": overall_best_genome,
                "prey_genome": prey_genome,
                "fitness_history": best_fitness_history
            },
            f,
            indent=2
        )


if __name__ == "__main__":
    evolve()