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

def create_lookup_dicts(datacenters, servers, selling_prices):
    # Create dictionaries for faster lookups
    datacenter_dict = datacenters.set_index('datacenter_id').to_dict('index')
    server_dict = servers.set_index('server_generation').to_dict('index')
    selling_price_dict = selling_prices.set_index(['server_generation', 'latency_sensitivity'])['selling_price'].to_dict()
    
    return datacenter_dict, server_dict, selling_price_dict

def solve_fleet_optimization(demand: pd.DataFrame, 
                             datacenter_dict: Dict, 
                             server_dict: Dict, 
                             selling_price_dict: Dict, 
                             time_step: int, 
                             current_fleet: Dict[str, Dict[str, Tuple[str, int]]],
                             ) -> Tuple[List[Dict], Dict[str, Dict[str, Tuple[str, int]]], float, float, float]:
    
    prob = pulp.LpProblem(f"Server_Fleet_Management_Step_{time_step}", pulp.LpMaximize)

    datacenter_ids = list(datacenter_dict.keys())
    server_generations = list(server_dict.keys())

    # Decision Variables
    buy = {(d, s): pulp.LpVariable(f"buy_{d}_{s}", lowBound=0, cat='Integer')
           for d in datacenter_ids for s in server_generations}
    
    move = {(d1, d2, s): pulp.LpVariable(f"move_{d1}_{d2}_{s}", lowBound=0, cat='Integer')
            for d1 in datacenter_ids for d2 in datacenter_ids 
            if d1 != d2 for s in server_generations}
    
    dismiss = {(d, s): pulp.LpVariable(f"dismiss_{d}_{s}", lowBound=0, cat='Integer')
               for d in datacenter_ids for s in server_generations}

    server_count = {(d, s): pulp.LpVariable(f"count_{d}_{s}", lowBound=0, cat='Integer')
                    for d in datacenter_ids for s in server_generations}

    # Precompute total demand
    numeric_columns = demand.select_dtypes(include=[np.number]).columns
    total_demand = demand[numeric_columns].sum().sum()

    # Objective function components
    utilized_capacity = pulp.LpVariable("utilized_capacity", lowBound=0)
    profit = pulp.LpVariable("profit")
    lifespan_sum = pulp.LpVariable("lifespan_sum", lowBound=0)

    EXPECTED_FAILURE_RATE = 0.075

    # Expected capacity for optimization
    expected_total_capacity = pulp.lpSum(
        server_count[d, s] * server_dict[s]['capacity'] * (1 - EXPECTED_FAILURE_RATE)
        for d in datacenter_ids for s in server_generations
    )

    normalized_lifespan = pulp.LpVariable("normalized_lifespan", lowBound=0, upBound=1)
    
    total_servers = pulp.lpSum(server_count[d, s] for d in datacenter_ids for s in server_generations)
    M = 1000000
    prob += lifespan_sum - M * (1 - normalized_lifespan) <= total_servers
    prob += lifespan_sum + M * (1 - normalized_lifespan) >= total_servers
    prob += total_servers >= 1

    # Normalization factors
    max_utilized_capacity = total_demand
    max_lifespan = 1.0
    max_profit = 1000000

    scaled_utilization = pulp.LpVariable("scaled_utilization", lowBound=0)
    scaled_lifespan = pulp.LpVariable("scaled_lifespan", lowBound=0)
    scaled_profit = pulp.LpVariable("scaled_profit", lowBound=0)

    prob += scaled_utilization * max_utilized_capacity == utilized_capacity 
    prob += scaled_lifespan * max_lifespan == normalized_lifespan
    prob += scaled_profit * max_profit == profit

    # Objective
    utilization_weight, lifespan_weight, profit_weight = 100, 10, 1
    prob += utilization_weight * scaled_utilization + lifespan_weight * scaled_lifespan + profit_weight * scaled_profit, "Objective"

    # Constraints
    for d in datacenter_ids:
        prob += pulp.lpSum(server_count[d, s] * server_dict[s]['slots_size']
                           for s in server_generations) <= datacenter_dict[d]['slots_capacity'], f"Slots_Capacity_{d}"

    for d in datacenter_ids:
        for s in server_generations:
            current_count = sum(1 for server_gen, _ in current_fleet[d].values() if server_gen == s)
            prob += server_count[d, s] == (
                current_count + buy[d, s] +
                pulp.lpSum(move[d2, d, s] for d2 in datacenter_ids if d2 != d) -
                pulp.lpSum(move[d, d2, s] for d2 in datacenter_ids if d2 != d) -
                dismiss[d, s]
            )

    for s in server_generations:
        release_time = eval(server_dict[s]['release_time'])
        if time_step < release_time[0] or time_step > release_time[1]:
            for d in datacenter_ids:
                prob += buy[d, s] == 0

    prob += utilized_capacity <= expected_total_capacity
    prob += utilized_capacity <= total_demand

    for d in datacenter_ids:
        for s in server_generations:
            life_expectancy = server_dict[s]['life_expectancy']
            for server_id, (server_gen, purchase_time) in list(current_fleet[d].items()):
                if server_gen == s:
                    if time_step - purchase_time >= life_expectancy:
                        dismiss_count = int(dismiss[d, s].value() or 0)
                        prob += dismiss_count >= 1
                        prob += server_count[d, s] >= dismiss_count

    # Calculate profit
    revenue = pulp.lpSum(
        utilized_capacity * selling_price_dict[(s, datacenter_dict[d]['latency_sensitivity'])]
        for d in datacenter_ids for s in server_generations
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

    prob += profit == revenue - costs

    # Solve the problem
    solver = pulp.CPLEX_CMD(path=r"C:\Program Files\IBM\ILOG\CPLEX_Studio_Community2211\cplex\bin\x64_win64\cplex.exe", msg=True,threads= 4)
    prob.solve(solver)
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        return None, current_fleet, 0, 0, 0

    # Extract actions
    actions = []
    new_fleet = {d: {} for d in datacenter_ids}

    for d in datacenter_ids:
        for s in server_generations:
            buy_count = int(buy[d, s].value() or 0)
            for _ in range(buy_count):
                server_id = generate_server_id()
                actions.append({
                    "time_step": time_step,
                    "datacenter_id": d,
                    "server_id": server_id,
                    "server_generation": s,
                    "action": "buy"
                })
                new_fleet[d][server_id] = (s, time_step)
            
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
            
            dismissed_servers = []
            servers_to_dismiss = []
            for server_id, (server_gen, purchase_time) in list(current_fleet[d].items()):
                if server_gen == s:
                    current_age = time_step - purchase_time
                    life_expectancy = server_dict[s]['life_expectancy']
                    if current_age >= life_expectancy:
                        servers_to_dismiss.append(server_id)

            dismiss_count = int(dismiss[d, s].value() or 0)
            additional_dismissals = [
                server_id for server_id, (server_gen, _) in current_fleet[d].items()
                if server_gen == s and server_id not in servers_to_dismiss
            ][:dismiss_count - len(servers_to_dismiss)]
            servers_to_dismiss.extend(additional_dismissals)

            for server_id in servers_to_dismiss:
                if server_id in current_fleet[d]:
                    actions.append({
                        "time_step": time_step,
                        "datacenter_id": d,
                        "server_id": server_id,
                        "server_generation": s,
                        "action": "dismiss"
                    })
                    del current_fleet[d][server_id]
                    dismissed_servers.append(server_id)
            
            for server_id, (server_gen, purchase_time) in current_fleet[d].items():
                if server_gen == s and server_id not in dismissed_servers:
                    new_fleet[d][server_id] = (server_gen, purchase_time)
    
    # Calculate actual utilization and normalized lifespan
    total_capacity = sum(
        adjust_capacity_by_failure_rate(
            server_count[d, s].value() * server_dict[s]['capacity']
        )
        for d in datacenter_ids 
        for s in server_generations
    )
    utilization = min(utilized_capacity.value() / total_capacity, 1.0) if total_capacity > 0 else 0

    total_servers = sum(len(servers) for servers in new_fleet.values())
    if total_servers > 0:
        actual_lifespan = sum(
            (time_step - purchase_time) / server_dict[server_gen]['life_expectancy']
            for d in new_fleet
            for server_id, (server_gen, purchase_time) in new_fleet[d].items()
        ) / total_servers
    else:
        actual_lifespan = 0

    return actions, new_fleet, utilization, actual_lifespan, profit.value()

def solve_multi_time_steps(actual_demand: pd.DataFrame, 
                           datacenters: pd.DataFrame, 
                           servers: pd.DataFrame, 
                           selling_prices: pd.DataFrame, 
                           total_time_steps: int = 168) -> Tuple[List[Dict], List[Dict]]:
    all_actions = []
    results = []
    
    # Create lookup dictionaries
    datacenter_dict, server_dict, selling_price_dict = create_lookup_dicts(datacenters, servers, selling_prices)
    
    current_fleet = {d: {} for d in datacenter_dict.keys()}
    
    for time_step in range(1, total_time_steps + 1):
        print(f"Solving for time step {time_step}")
        time_step_demand = actual_demand[actual_demand['time_step'] == time_step]
        
        result = solve_fleet_optimization(
            time_step_demand, datacenter_dict, server_dict, selling_price_dict, time_step, current_fleet
        )
        
        if result is None:
            print(f"Failed to find a solution for time step {time_step}")
            continue
        
        actions, current_fleet, utilization, lifespan, profit = result
        all_actions.extend(actions)
        print(f"Utilization: {utilization:.2f}")
        print(f"Lifespan: {lifespan:.2f}")
        print(f"Profit: {profit:.2f}")
        print(f"Total servers: {sum(len(servers) for servers in current_fleet.values())}")
        results.append(
            {
                "time_step": time_step,
                "utilization": utilization,
                "lifespan": lifespan,
                "profit": profit,
                "total_servers": sum(len(servers) for servers in current_fleet.values())
            }
        )

    return all_actions, results

def main():
    base_demand, datacenters, servers, selling_prices = load_problem_data()
    seeds = known_seeds('training')

    np.random.seed(3163)
    
    actual_demand = get_actual_demand(base_demand)
    
    solution, results = solve_multi_time_steps(actual_demand, datacenters, servers, selling_prices)
    if solution:
        save_solution(solution, f"solution_{3163}.json")
        save_solution(results, f"results_{3163}.json")
        print(f"Solution for seed {3163} saved to 'solution_{3163}.json'")
    else:
        print(f"Failed to find a solution for seed {3163}")

if __name__ == "__main__":
    main()

    #return the the fleet at the end