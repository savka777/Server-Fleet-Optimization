import os
from evaluation import evaluation_function
from utils import load_solution, load_problem_data

# Path to the directory where your solutions are stored
solution_directory = './'

# LOAD PROBLEM DATA (once, since the data remains the same)
demand, datacenters, servers, selling_prices = load_problem_data()

# List all JSON solution files in the directory
solution_files = [f for f in os.listdir(solution_directory) if f.endswith('.json')]

# Loop through each solution file and evaluate
for solution_file in solution_files:
    # Extract seed number from the file name (e.g., '8761.json')
    seed = int(solution_file.split('.')[0])

    # Load the solution
    solution = load_solution(os.path.join(solution_directory, solution_file))

    # Evaluate the solution
    score = evaluation_function(solution,
                                demand,
                                datacenters,
                                servers,
                                selling_prices,
                                seed=seed,
                                verbose=1)

    # Print the seed and score
    print(f'Seed: {seed}, Solution score: {score}')
