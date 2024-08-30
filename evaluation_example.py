

from utils import (load_problem_data,
                   load_solution)
from evaluation_v6 import evaluation_function


# LOAD SOLUTION
solution = load_solution('./multi_time_step_solution.json')

# LOAD PROBLEM DATA
demand, datacenters, servers, selling_prices = load_problem_data()

# EVALUATE THE SOLUTION
score = evaluation_function(solution,
                            demand,
                            datacenters,
                            servers,
                            selling_prices,
                            seed=None)

print(f'Solution score: {score}')