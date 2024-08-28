import pulp
import pandas as pd
import numpy as np
from utils import load_problem_data, save_solution
from evaluation import get_actual_demand

def solve_single_time_step(demand, datacenters, servers, selling_prices, time_step, current_fleet):
    prob = pulp.LpProblem(f"Server_Fleet_Management_Step_{time_step}", pulp.LpMaximize)

    datacenter_ids = datacenters['datacenter_id'].tolist()
    server_types = servers['server_generation'].tolist()
    latency_sensitivities = ['high', 'medium', 'low']

    # Decision variables
    buy = {(d, s): pulp.LpVariable(f"buy_{time_step}_{d}_{s}", lowBound=0, cat='Integer')
           for d in datacenter_ids for s in server_types}
    
    move = {(d1, d2, s): pulp.LpVariable(f"move_{time_step}_{d1}_{d2}_{s}", lowBound=0, cat='Integer')
            for d1 in datacenter_ids for d2 in datacenter_ids if d1 != d2 for s in server_types}
    
    hold = {(d, s): pulp.LpVariable(f"hold_{time_step}_{d}_{s}", lowBound=0, cat='Integer')
            for d in datacenter_ids for s in server_types}
    
    dismiss = {(d, s): pulp.LpVariable(f"dismiss_{time_step}_{d}_{s}", lowBound=0, cat='Integer')
               for d in datacenter_ids for s in server_types}

    # Server count variables
    server_count = {(d, s): pulp.LpVariable(f"count_{time_step}_{d}_{s}", lowBound=0, cat='Integer')
                    for d in datacenter_ids for s in server_types}

    # Add slack variables
    capacity_slack = {d: pulp.LpVariable(f"capacity_slack_{d}", lowBound=0) for d in datacenter_ids}
    demand_slack = {s: pulp.LpVariable(f"demand_slack_{s}", lowBound=0) for s in server_types}

    # Calculate demand for this time step
    time_step_demand = demand[demand['time_step'] == time_step]
    total_demand = {s: time_step_demand[time_step_demand['server_generation'] == s][latency_sensitivities].sum().sum() 
                    for s in server_types}
    total_demand_all = sum(total_demand.values())

    # Capacity calculation
    total_capacity = pulp.lpSum(server_count[d, s] * servers[servers['server_generation'] == s]['capacity'].values[0]
                                for d in datacenter_ids for s in server_types)

    # Utilization constraint
    prob += total_capacity >= total_demand_all

    # Lifespan component (L)
    total_servers = pulp.lpSum(server_count[d, s] for d in datacenter_ids for s in server_types)
    total_lifespan = pulp.lpSum(
        (current_fleet.get((d, s), 0) * (time_step - 1) + buy[d, s]) * servers[servers['server_generation'] == s]['life_expectancy'].values[0]
        for d in datacenter_ids for s in server_types
    ) if current_fleet is not None else pulp.lpSum(buy[d, s] * servers[servers['server_generation'] == s]['life_expectancy'].values[0]
                                                   for d in datacenter_ids for s in server_types)

    # Objective function components
    revenue = pulp.lpSum(server_count[d, s] * servers[servers['server_generation'] == s]['capacity'].values[0] *
                         selling_prices[(selling_prices['server_generation'] == s) & 
                                        (selling_prices['latency_sensitivity'] == datacenters[datacenters['datacenter_id'] == d]['latency_sensitivity'].values[0])]['selling_price'].values[0]
                         for d in datacenter_ids for s in server_types)

    purchase_cost = pulp.lpSum(buy[d, s] * servers[servers['server_generation'] == s]['purchase_price'].values[0]
                               for d in datacenter_ids for s in server_types)

    energy_cost = pulp.lpSum(server_count[d, s] * servers[servers['server_generation'] == s]['energy_consumption'].values[0] *
                             datacenters[datacenters['datacenter_id'] == d]['cost_of_energy'].values[0]
                             for d in datacenter_ids for s in server_types)

    move_cost = pulp.lpSum(move[d1, d2, s] * servers[servers['server_generation'] == s]['cost_of_moving'].values[0]
                           for d1 in datacenter_ids for d2 in datacenter_ids if d1 != d2 for s in server_types)

    def maintenance_cost(s):
        b = servers[servers['server_generation'] == s]['average_maintenance_fee'].values[0]
        x_hat = servers[servers['server_generation'] == s]['life_expectancy'].values[0]
        return b * (1 + ((1.5 * time_step) / x_hat) * np.log2((1.5 * time_step) / x_hat))

    maintenance_cost = pulp.lpSum(server_count[d, s] * maintenance_cost(s)
                                  for d in datacenter_ids for s in server_types)
    
    profit = revenue - purchase_cost - energy_cost - move_cost - maintenance_cost

    # Objective: Maximize a combination of profit, utilization (represented by minimizing excess capacity), and lifespan
    prob += profit - 0.1 * (total_capacity - total_demand_all) + total_lifespan / 1000 - 1000000 * pulp.lpSum(capacity_slack.values()) - 1000000 * pulp.lpSum(demand_slack.values())

    # Constraints
    # Server count constraint
    for d in datacenter_ids:
        for s in server_types:
            if current_fleet is None:
                prob += server_count[d, s] == buy[d, s]
            else:
                current_count = current_fleet.get((d, s), 0)
                prob += server_count[d, s] == (
                    current_count + buy[d, s] +
                    pulp.lpSum(move[d2, d, s] for d2 in datacenter_ids if d2 != d) -
                    pulp.lpSum(move[d, d2, s] for d2 in datacenter_ids if d2 != d) -
                    dismiss[d, s]
                )
                prob += hold[d, s] == current_count - dismiss[d, s] - pulp.lpSum(move[d, d2, s] for d2 in datacenter_ids if d2 != d)

    # Modified demand satisfaction constraint
    for s in server_types:
        prob += (pulp.lpSum(server_count[d, s] * servers[servers['server_generation'] == s]['capacity'].values[0]
                           for d in datacenter_ids) 
                 >= total_demand[s] - demand_slack[s])

    # Modified datacenter slot capacity constraint
    for d in datacenter_ids:
        prob += (pulp.lpSum(server_count[d, s] * servers[servers['server_generation'] == s]['slots_size'].values[0]
                           for s in server_types) 
                 <= datacenters[datacenters['datacenter_id'] == d]['slots_capacity'].values[0] + capacity_slack[d])

    # Server release time constraint
    for s in server_types:
        release_time = eval(servers[servers['server_generation'] == s]['release_time'].values[0])
        if time_step < release_time[0] or time_step > release_time[1]:
            for d in datacenter_ids:
                prob += buy[d, s] == 0

     # Solve the problem with an optimality gap and time limit
    solver = pulp.PULP_CBC_CMD(msg=1, gapRel=0.01, timeLimit=600)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] == 'Optimal':
        print(f"Optimal solution found for time step {time_step}")
    elif pulp.LpStatus[prob.status] == 'Not Solved':
        print(f"Warning: Problem not solved for time step {time_step}. Status: {pulp.LpStatus[prob.status]}")
        return None
    else:
        print(f"Solution found for time step {time_step}, but may not be optimal. Status: {pulp.LpStatus[prob.status]}")

    # Calculate and print the optimality gap
    if hasattr(solver, 'mipGap'):
        gap = solver.mipGap
        print(f"Optimality gap: {gap:.2%}")
    else:
        print("Optimality gap information not available")

    # Extract actions and calculate metrics
    actions = []
    for d in datacenter_ids:
        for s in server_types:
            buy_count = int(buy[d, s].varValue or 0)
            for i in range(buy_count):
                actions.append({
                    "time_step": time_step,
                    "datacenter_id": d,
                    "server_id": f"{s}_{d}_{time_step}_{i+1}",
                    "server_generation": s,
                    "action": "buy"
                })
            
            for d2 in datacenter_ids:
                if d != d2:
                    move_count = int(move[d, d2, s].varValue or 0)
                    for i in range(move_count):
                        actions.append({
                            "time_step": time_step,
                            "datacenter_id": d2,
                            "server_id": f"{s}_{d}_{time_step}_{i+1}",
                            "server_generation": s,
                            "action": "move"
                        })
            
            dismiss_count = int(dismiss[d, s].varValue or 0)
            for i in range(dismiss_count):
                actions.append({
                    "time_step": time_step,
                    "datacenter_id": d,
                    "server_id": f"{s}_{d}_{time_step}_{i+1}",
                    "server_generation": s,
                    "action": "dismiss"
                })

    current_fleet = {(d, s): server_count[d, s].varValue for d in datacenter_ids for s in server_types}
    utilization = total_demand_all / total_capacity.value()
    lifespan = total_lifespan.value() / (total_servers.value() * 1000) if total_servers.value() > 0 else 0

    print(f"Utilization: {utilization:.2f}")
    print(f"Lifespan: {lifespan:.2f}")
    print(f"Profit: {profit.value():.2f}")

    # Print slack variable values
    print("Slack variable values:")
    for d in datacenter_ids:
        if capacity_slack[d].varValue > 0:
            print(f"Capacity slack for {d}: {capacity_slack[d].varValue}")
    for s in server_types:
        if demand_slack[s].varValue > 0:
            print(f"Demand slack for {s}: {demand_slack[s].varValue}")

    return actions, current_fleet, utilization, lifespan, profit.value()

def solve_multi_time_steps(demand, datacenters, servers, selling_prices, total_time_steps=168, horizon=30):
    all_actions = []
    current_fleet = None

    for horizon_start in range(1, total_time_steps + 1, horizon):
        horizon_end = min(horizon_start + horizon - 1, total_time_steps)
        print(f"\n{'='*50}")
        print(f"Solving horizon: Time steps {horizon_start} to {horizon_end}")
        print(f"{'='*50}")

        horizon_actions = []
        for time_step in range(horizon_start, horizon_end + 1):
            print(f"\nSolving for time step {time_step}")
            result = solve_single_time_step(
                demand, datacenters, servers, selling_prices, time_step, current_fleet
            )

            if result is None:
                print(f"Failed to find a solution for time step {time_step}")
                print("Using previous fleet configuration and continuing to next time step")
                continue

            solution, current_fleet, utilization, lifespan, profit = result
            horizon_actions.extend(solution)

            print(f"Utilization: {utilization:.2f}")
            print(f"Lifespan: {lifespan:.2f}")
            print(f"Profit: {profit:.2f}")

        all_actions.extend(horizon_actions)

        print(f"\nCompleted horizon: Time steps {horizon_start} to {horizon_end}")
        print(f"Total actions so far: {len(all_actions)}")

        # Optionally, you can save intermediate results here
        # save_solution(all_actions, f"solution_up_to_step_{horizon_end}.json")

    return all_actions

def main():
    # Load problem data
    demand, datacenters, servers, selling_prices = load_problem_data()

    # Get actual demand
    actual_demand = get_actual_demand(demand)

    # Solve for all time steps
    solution = solve_multi_time_steps(actual_demand, datacenters, servers, selling_prices)

    if solution:
        # Save the solution
        save_solution(solution, "multi_time_step_solution.json")
        print("Solution saved to 'multi_time_step_solution.json'")
        
        # Print summary
        action_counts = {"buy": 0, "move": 0, "dismiss": 0}
        server_counts = {}
        for action in solution:
            action_type = action['action']
            server_type = action['server_generation']
            action_counts[action_type] += 1
            if action_type == "buy":
                server_counts[server_type] = server_counts.get(server_type, 0) + 1
        
        print("\nTotal actions taken:")
        for action_type, count in action_counts.items():
            print(f"{action_type}: {count}")
        
        print("\nTotal servers purchased:")
        for server_type, count in server_counts.items():
            print(f"{server_type}: {count}")
        
        print(f"\nTotal actions: {len(solution)}")
    else:
        print("Failed to find any solutions")

if __name__ == "__main__":
    main()