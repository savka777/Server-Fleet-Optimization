import uuid
import pulp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from evaluation import get_actual_demand, adjust_capacity_by_failure_rate, get_maintenance_cost
from seeds import known_seeds
from utils import load_problem_data, save_solution

def generate_server_id():
    return str(uuid.uuid4())

# Dataframes are to slow for looks up use this instead
def create_lookup_dicts(datacenters, servers, selling_prices):
    datacenter_dict = datacenters.set_index('datacenter_id').to_dict('index')
    server_dict = servers.set_index('server_generation').to_dict('index')
    selling_price_dict = selling_prices.set_index(['server_generation', 'latency_sensitivity'])['selling_price'].to_dict()
    
    return datacenter_dict, server_dict, selling_price_dict

def get_maintenance_cost(base_cost, current_time, life_expectancy):
    return base_cost * (1 + (1.5 * current_time / life_expectancy) * np.log2(1.5 * current_time / life_expectancy))

def solve_fleet_optimization(demand: Dict, 
                             datacenter_dict: Dict, 
                             server_dict: Dict, 
                             selling_price_dict: Dict, 
                             time_step: int, 
                             current_fleet: Dict[str, Dict[str, Tuple[str, int]]]) -> Tuple[List[Dict], Dict[str, Dict[str, Tuple[str, int]]], float, float, float]:
    prob = pulp.LpProblem(f"Server_Fleet_Management_Step_{time_step}", pulp.LpMaximize)
    # Server ID tracking
    server_id_counter = max([int(server_id.split('-')[-1]) for dc in current_fleet.values() for server_id in dc.keys()] + [0])
    def get_next_server_id():
      nonlocal server_id_counter
      server_id_counter += 1
      return f"server-{server_id_counter:04d}"

    datacenter_ids = list(datacenter_dict.keys())
    server_generations = list(server_dict.keys())
    latency_sensitivities = ['high', 'medium', 'low']

    '''Decision Variables'''
    buy = {(d, s): pulp.LpVariable(f"buy_{d}_{s}", lowBound=0, cat='Integer')
           for d in datacenter_ids for s in server_generations}
    
    move = {(d1, d2, s): pulp.LpVariable(f"move_{d1}_{d2}_{s}", lowBound=0, cat='Integer')
            for d1 in datacenter_ids for d2 in datacenter_ids 
            if d1 != d2 for s in server_generations}
    
    dismiss = {(d, s): pulp.LpVariable(f"dismiss_{d}_{s}", lowBound=0, cat='Integer')
               for d in datacenter_ids for s in server_generations}

    server_count = {(d, s): pulp.LpVariable(f"count_{d}_{s}", lowBound=0, cat='Integer')
                    for d in datacenter_ids for s in server_generations}

    demand_met = {(d, s, l): pulp.LpVariable(f"demand_met_{d}_{s}_{l}", lowBound=0)
                  for d in datacenter_ids for s in server_generations for l in latency_sensitivities}

    # Auxiliary variables for optimization goals
    total_demand_met = pulp.lpSum(demand_met[d, s, l] for d in datacenter_ids for s in server_generations for l in latency_sensitivities)
    
    total_age = pulp.lpSum(
        server_count[d, s] * (time_step - current_fleet[d].get(s, (None, time_step))[1])
        for d in datacenter_ids for s in server_generations
    )
    
    profit = pulp.LpVariable("profit")

    '''Constraints'''
    # 1. Data Center Capacity
    for d in datacenter_ids:
        prob += pulp.lpSum(server_count[d, s] * server_dict[s]['slots_size']
                           for s in server_generations) <= datacenter_dict[d]['slots_capacity'], f"Slots_Capacity_{d}"

    # 2. Demand Fulfillment
    for d in datacenter_ids:
        for s in server_generations:
            for l in latency_sensitivities:
                if (s, l) in demand:
                    prob += demand_met[d, s, l] <= demand[s, l], f"Demand_Fulfillment_{d}_{s}_{l}"

    # 3. Server Count Dynamics
    for d in datacenter_ids:
        for s in server_generations:
            current_count = sum(1 for server_gen, _ in current_fleet[d].values() if server_gen == s)
            prob += server_count[d, s] == (
                current_count + buy[d, s] +
                pulp.lpSum(move[d2, d, s] for d2 in datacenter_ids if d2 != d) -
                dismiss[d, s] -
                pulp.lpSum(move[d, d2, s] for d2 in datacenter_ids if d2 != d)
            ), f"Server_Count_Dynamics_{d}_{s}"

    # 4. Lifecycle Management
    for d in datacenter_ids:
        for s in server_generations:
            life_expectancy = server_dict[s]['life_expectancy']
            prob += dismiss[d, s] >= pulp.lpSum(
                1 for server_id, (server_gen, purchase_time) in current_fleet[d].items()
                if server_gen == s and time_step - purchase_time >= life_expectancy
            ), f"Lifecycle_Management_{d}_{s}"

    # 5. Purchase Window
    for s in server_generations:
        release_time = eval(server_dict[s]['release_time'])
        if time_step < release_time[0] or time_step > release_time[1]:
            for d in datacenter_ids:
                prob += buy[d, s] == 0, f"Purchase_Window_{d}_{s}"

    # 6. Utilization
    EXPECTED_FAILURE_RATE = 0.075
    total_capacity = pulp.lpSum(
        server_count[d, s] * server_dict[s]['capacity'] * (1 - EXPECTED_FAILURE_RATE)
        for d in datacenter_ids for s in server_generations
    )
    prob += total_demand_met <= total_capacity, "Utilization_Constraint"

    # 7. Profit Calculation
    revenue = pulp.lpSum(
        demand_met[d, s, l] * selling_price_dict[(s, l)]
        for d in datacenter_ids for s in server_generations for l in latency_sensitivities
    )

    costs = pulp.lpSum(
        buy[d, s] * server_dict[s]['purchase_price'] +
        server_count[d, s] * (
            server_dict[s]['energy_consumption'] * 
            datacenter_dict[d]['cost_of_energy'] +
            get_maintenance_cost(
                server_dict[s]['average_maintenance_fee'],
                time_step,
                server_dict[s]['life_expectancy']
            )
        ) +
        pulp.lpSum(move[d, d2, s] * server_dict[s]['cost_of_moving']
                   for d2 in datacenter_ids if d2 != d)
        for d in datacenter_ids for s in server_generations
    )

    prob += profit == revenue - costs, "Profit_Calculation"

    # Objective: Maximize a weighted sum of demand met, server age, and profit
    w1, w2, w3 = 0.4, 0.3, 0.3
    prob += w1 * total_demand_met + w2 * total_age + w3 * profit, "Objective"

    # Solver
    # solver = pulp.PULP_CBC_CMD(msg=1, gapRel=0.10,  threads=4) # Use if CPLEX is not installed
    solver = pulp.CPLEX_CMD(path=r"C:\Program Files\IBM\ILOG\CPLEX_Studio_Community2211\cplex\bin\x64_win64\cplex.exe", msg=True, threads=4)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != 'Optimal':
        print(f"Failed to find an optimal solution for time step {time_step}")
        return None, current_fleet, 0, 0, 0

    # Extract actions and update fleet
    actions, new_fleet = extract_actions_and_update_fleet(buy, move, dismiss, server_count, current_fleet, datacenter_ids, server_generations, time_step, get_next_server_id)

    # Calculate actual values for U, L, and P
    actual_utilization = total_demand_met.value() / (total_capacity.value() + 1e-6)
    actual_lifespan = total_age.value() / (pulp.lpSum(server_count[d, s] for d in datacenter_ids for s in server_generations).value() + 1e-6)
    actual_profit = profit.value()

    # Debug print
    print(f"Time step {time_step}:")
    print(f"  Utilization (U): {actual_utilization:.2f}")
    print(f"  Lifespan (L): {actual_lifespan:.2f}")
    print(f"  Profit (P): {actual_profit:.2f}")

    return actions, new_fleet, actual_utilization, actual_lifespan, actual_profit

def extract_actions_and_update_fleet(buy, move, dismiss, server_count, current_fleet, datacenter_ids, server_generations, time_step, get_next_server_id):
    actions = []
    new_fleet = {d: {} for d in datacenter_ids}

    for d in datacenter_ids:
        for s in server_generations:
            # Buy actions
            buy_count = int(buy[d, s].value() or 0)
            for _ in range(buy_count):
                server_id = get_next_server_id()
                actions.append({
                    "time_step": time_step,
                    "datacenter_id": d,
                    "server_id": server_id,
                    "server_generation": s,
                    "action": "buy"
                })
                new_fleet[d][server_id] = (s, time_step)
            
            # Move actions
            for d2 in datacenter_ids:
                if d != d2:
                    move_count = int(move[d, d2, s].value() or 0)
                    moved_servers = []
                    for server_id, (server_gen, purchase_time) in list(current_fleet[d].items())[:move_count]:
                        if server_gen == s:
                            actions.append({
                                "time_step": time_step,
                                "datacenter_id": d2,
                                "server_id": server_id,
                                "server_generation": s,
                                "action": "move"
                            })
                            new_fleet[d2][server_id] = (s, purchase_time)
                            moved_servers.append(server_id)
                    for server_id in moved_servers:
                        del current_fleet[d][server_id]
            
            # Dismiss actions
            dismiss_count = int(dismiss[d, s].value() or 0)
            dismissed_servers = []
            for server_id, (server_gen, purchase_time) in list(current_fleet[d].items()):
                if server_gen == s and len(dismissed_servers) < dismiss_count:
                    actions.append({
                        "time_step": time_step,
                        "datacenter_id": d,
                        "server_id": server_id,
                        "server_generation": s,
                        "action": "dismiss"
                    })
                    dismissed_servers.append(server_id)
            for server_id in dismissed_servers:
                del current_fleet[d][server_id]
            
            # Update fleet with remaining servers
            for server_id, (server_gen, purchase_time) in current_fleet[d].items():
                if server_gen == s and server_id not in dismissed_servers:
                    new_fleet[d][server_id] = (server_gen, purchase_time)

    return actions, new_fleet

def solve_multi_time_steps(actual_demand, datacenters, servers, selling_prices, total_time_steps=168):
    all_actions = []
    results = []
    
    datacenter_dict, server_dict, selling_price_dict = create_lookup_dicts(datacenters, servers, selling_prices)
    
    current_fleet = {d: {} for d in datacenter_dict.keys()}

    for time_step in range(1, total_time_steps + 1):
        print(f"Solving for time step {time_step}")
        time_step_demand = actual_demand[actual_demand['time_step'] == time_step]
        
        # Convert time_step_demand to the format expected by solve_fleet_optimization
        demand_dict = {}
        for _, row in time_step_demand.iterrows():
            server_gen = row['server_generation']
            for latency in ['high', 'medium', 'low']:
                demand_dict[(server_gen, latency)] = row[latency]
        
        result = solve_fleet_optimization(
            demand_dict, datacenter_dict, server_dict, selling_price_dict, time_step, current_fleet
        )
        
        if result is None:
            print(f"Failed to find a solution for time step {time_step}")
            continue
        
        actions, current_fleet, utilization, lifespan, profit = result
        all_actions.extend(actions)
        
        print(f"Time step {time_step} results:")
        print(f"  Actions taken: {len(actions)}")
        print(f"  Current fleet size: {sum(len(servers) for servers in current_fleet.values())}")
        print(f"  Utilization: {utilization:.2f}")
        print(f"  Lifespan: {lifespan:.2f}")
        print(f"  Profit: {profit:.2f}")
        
        results.append({
            "time_step": time_step,
            "utilization": utilization,
            "lifespan": lifespan,
            "profit": profit,
            "total_servers": sum(len(servers) for servers in current_fleet.values())
        })

    return all_actions, results

def main():

    base_demand, datacenters, servers, selling_prices = load_problem_data()

    all_seeds = known_seeds('test')

    all_results = {}

    for seed in all_seeds:
        try:
            print(f"\nUsing random seed: {seed}")
            np.random.seed(seed)
            
            actual_demand = get_actual_demand(base_demand)
        
            print("Starting multi-time step optimization...")
            solution, results = solve_multi_time_steps(actual_demand, datacenters, servers, selling_prices)
        
            if solution:
                solution_file = f"{seed}.json"
                save_solution(solution, solution_file)
                print(f"Solution saved to '{solution_file}'")
                
                print("\nOptimization Summary:")
                print(f"Total time steps: {len(results)}")
                print(f"Final utilization: {results[-1]['utilization']:.2f}")
                print(f"Final lifespan: {results[-1]['lifespan']:.2f}")
                print(f"Final profit: {results[-1]['profit']:.2f}")
                print(f"Final total servers: {results[-1]['total_servers']}")
            else:
                print(f"Failed to find a solution for seed {seed}")
        
        except Exception as e:
            print(f"An error occurred during optimization for seed {seed}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            all_results[seed] = None

    print("\nOverall Summary:")
    for seed, result in all_results.items():
        if result:
            print(f"Seed {seed}:")
            print(f"  Utilization: {result['utilization']:.2f}")
            print(f"  Lifespan: {result['lifespan']:.2f}")
            print(f"  Profit: {result['profit']:.2f}")
            print(f"  Total servers: {result['total_servers']}")
        else:
            print(f"Seed {seed}: Failed to find a solution")

if __name__ == "__main__":
    main()