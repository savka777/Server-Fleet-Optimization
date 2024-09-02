import uuid
import pulp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from evaluation_v6 import get_actual_demand, adjust_capacity_by_failure_rate, get_maintenance_cost
from utils import load_problem_data, save_solution
import logging

import matplotlib.pyplot as plt


# Generate a unique ID for new servers
def generate_server_id():
    return str(uuid.uuid4())


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Constraint for max purchases, might not need. Removed for now but see line 131
def calculate_max_purchase(demand: pd.DataFrame, servers: pd.DataFrame) -> int:
    """
    Calculate the maximum number of servers that might need to be purchased in a single time step.
    This is based on the maximum demand across all time steps and the minimum server capacity.
    """
    actual_demand = get_actual_demand(demand)

    print("Actual Demand DataFrame Structure:")
    print(actual_demand.head())
    print("\nColumns:", actual_demand.columns)
    print("\nData types:", actual_demand.dtypes)

    # Sum only the numeric columns (high, low, medium)
    numeric_columns = actual_demand.select_dtypes(include=[np.number]).columns
    total_demand_per_timestep = actual_demand.groupby('time_step')[numeric_columns].sum().sum(axis=1)
    max_total_demand = total_demand_per_timestep.max()

    print(f"\nMaximum total demand: {max_total_demand}")

    min_server_capacity = servers['capacity'].min()
    print(f"Minimum server capacity: {min_server_capacity}")

    max_purchase = int(np.ceil(max_total_demand / min_server_capacity))
    print(f"Calculated max purchase: {max_purchase}")

    return max_purchase


# Function to automatically dismiss servers that have exceeded their lifetime
def automatically_dismiss_servers(current_fleet: Dict[str, Dict[str, Tuple[str, int]]],
                                  servers: pd.DataFrame,
                                  time_step: int) -> Tuple[Dict[str, Dict[str, Tuple[str, int]]], List[Dict]]:
    dismissed_servers = []
    new_fleet = {d: {} for d in current_fleet}

    for datacenter, dc_servers in current_fleet.items():
        for server_id, (server_gen, purchase_time) in dc_servers.items():
            life_expectancy = servers[servers['server_generation'] == server_gen]['life_expectancy'].values[0]
            if time_step - purchase_time >= life_expectancy:  # Dismiss after 96 time steps
                dismissed_servers.append({
                    "time_step": time_step,
                    "datacenter_id": datacenter,
                    "server_id": server_id,
                    "server_generation": server_gen,
                    "action": "dismiss"
                })
            else:
                new_fleet[datacenter][server_id] = (server_gen, purchase_time)

    return new_fleet, dismissed_servers


# Function to forecast future demand (simplified)
def forecast_demand(demand: pd.DataFrame, time_step: int, lookahead: int = 5) -> pd.DataFrame:
    # Forecast future demand as an average of the next 'lookahead' steps
    future_demand = demand[demand['time_step'].between(time_step, time_step + lookahead)]
    return future_demand.groupby('latency_sensitivity').mean()


# Function to calculate the proportion of demand met for deployed capacity
def calculate_utilization(server_count: Dict[str, int], servers: pd.DataFrame, failure_rate: float) -> float:
    total_capacity = sum(
        server_count[s] * servers[servers['server_generation'] == s]['capacity'].values[0] * (1 - failure_rate)
        for s in server_count)
    return total_capacity


# Incremental optimization for a time step
def solve_fleet_optimization(demand: pd.DataFrame,
                             datacenters: pd.DataFrame,
                             servers: pd.DataFrame,
                             selling_prices: pd.DataFrame,
                             time_step: int,
                             current_fleet: Dict[str, Dict[str, str]],
                             max_purchase_per_step: int) -> Tuple[
    List[Dict], Dict[str, Dict[str, str]], float, float, float]:
    # Automatically dismiss servers that have reached their lifetime
    current_fleet, auto_dismissed = automatically_dismiss_servers(current_fleet, servers, time_step)

    # Forecast future demand
    future_demand = forecast_demand(demand, time_step)

    # Initialize optimization problem
    prob = pulp.LpProblem(f"Server_Fleet_Management_Step_{time_step}", pulp.LpMaximize)

    # Decision variables optimized to reduce variables
    buy = {(d, s): pulp.LpVariable(f"buy_{d}_{s}", lowBound=0, upBound=max_purchase_per_step, cat='Integer')
           for d in datacenters['datacenter_id'] for s in servers['server_generation']}

    move = {(d1, d2, s): pulp.LpVariable(f"move_{d1}_{d2}_{s}", lowBound=0, cat='Integer')
            for d1 in datacenters['datacenter_id'] for d2 in datacenters['datacenter_id']
            if d1 != d2 for s in servers['server_generation']}

    dismiss = {(d, s): pulp.LpVariable(f"dismiss_{d}_{s}", lowBound=0, cat='Integer')
               for d in datacenters['datacenter_id'] for s in servers['server_generation']}

    # Server count
    server_count = {(d, s): pulp.LpVariable(f"count_{d}_{s}", lowBound=0, cat='Integer')
                    for d in datacenters['datacenter_id'] for s in servers['server_generation']}

    # Actual demand for the time step
    actual_demand = get_actual_demand(demand)
    time_step_demand = actual_demand[actual_demand['time_step'] == time_step]

    # Objective Function: Maximizing utilized capacity, lifespan, and profit
    utilized_capacity = pulp.LpVariable("utilized_capacity", lowBound=0)
    lifespan_numerator = pulp.LpVariable("lifespan_numerator", lowBound=0)
    profit = pulp.LpVariable("profit")

    # Objective: Maximize a linear combination of Utilization, Lifespan, and Profit
    # Adjust weights to reflect the importance of each component
    utilization_weight = 100
    lifespan_weight = 10
    profit_weight = 1
    prob += utilization_weight * utilized_capacity + lifespan_weight * lifespan_numerator + profit_weight * profit, "Objective"

    # Slot Capacity Constraints
    for d in datacenters['datacenter_id']:
        prob += pulp.lpSum(server_count[d, s] * servers[servers['server_generation'] == s]['slots_size'].values[0]
                           for s in servers['server_generation']) <= \
                datacenters[datacenters['datacenter_id'] == d]['slots_capacity'].values[0], f"Slots_Capacity_{d}"

    # Incremental server update, using previous solution
    for d in datacenters['datacenter_id']:
        for s in servers['server_generation']:
            current_count = sum(1 for server_gen in current_fleet.get(d, {}).values() if server_gen == s)
            prob += server_count[d, s] == (
                    current_count + buy[d, s] +
                    pulp.lpSum(move[d2, d, s] for d2 in datacenters['datacenter_id'] if d2 != d) -
                    pulp.lpSum(move[d, d2, s] for d2 in datacenters['datacenter_id'] if d2 != d) -
                    dismiss[d, s]
            )

    # Release time constraints (optimized to avoid adding unnecessary variables)
    for s in servers['server_generation']:
        release_time = eval(servers[servers['server_generation'] == s]['release_time'].values[0])
        if time_step < release_time[0] or time_step > release_time[1]:
            for d in datacenters['datacenter_id']:
                prob += buy[d, s] == 0

    # Total Capacity and Demand
    average_failure_rate = 0.075
    total_capacity = pulp.lpSum(
        server_count[d, s] * servers[servers['server_generation'] == s]['capacity'].values[0] * (
                1 - average_failure_rate)
        for d in datacenters['datacenter_id'] for s in servers['server_generation'])
    numeric_columns = time_step_demand.select_dtypes(include=[np.number]).columns
    total_demand = time_step_demand[numeric_columns].sum().sum()

    # Utilization Constraints
    prob += utilized_capacity <= total_capacity
    prob += utilized_capacity <= total_demand

    # Calculation of server lifespan
    total_servers = pulp.lpSum(
        server_count[d, s] for d in datacenters['datacenter_id'] for s in servers['server_generation'])
    prob += lifespan_numerator == pulp.lpSum(
        server_count[d, s] * time_step / servers[servers['server_generation'] == s]['life_expectancy'].values[0]
        for d in datacenters['datacenter_id'] for s in servers['server_generation']
    )
    prob += lifespan_numerator <= total_servers

    # Profit Calculation
    revenue = pulp.lpSum(
        utilized_capacity * selling_prices[(selling_prices['server_generation'] == s) &
                                           (selling_prices['latency_sensitivity'] ==
                                            datacenters[datacenters['datacenter_id'] == d][
                                                'latency_sensitivity'].values[0])]['selling_price'].values[0]
        for d in datacenters['datacenter_id'] for s in servers['server_generation']
    )

    costs = pulp.lpSum(
        buy[d, s] * servers[servers['server_generation'] == s]['purchase_price'].values[0] +
        server_count[d, s] * (
                servers[servers['server_generation'] == s]['energy_consumption'].values[0] *
                datacenters[datacenters['datacenter_id'] == d]['cost_of_energy'].values[0] +
                get_maintenance_cost(
                    servers[servers['server_generation'] == s]['average_maintenance_fee'].values[0],
                    time_step,
                    servers[servers['server_generation'] == s]['life_expectancy'].values[0]
                )
        ) +
        pulp.lpSum(move[d, d2, s] * servers[servers['server_generation'] == s]['cost_of_moving'].values[0]
                   for d2 in datacenters['datacenter_id'] if d2 != d)
        for d in datacenters['datacenter_id'] for s in servers['server_generation']
    )

    prob += profit == revenue - costs

    # Solve the problem with optimized parameters
    solver = pulp.PULP_CBC_CMD(msg=True, gapRel=0.05, timeLimit=300)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != 'Optimal':
        logger.warning(
            f"Problem not solved optimally for time step {time_step}. Status: {pulp.LpStatus[prob.status]}")
        return None, current_fleet, 0, 0, 0

    # Extract Actions
    actions = auto_dismissed.copy()
    new_fleet = {d: {} for d in datacenters['datacenter_id']}
    for d in datacenters['datacenter_id']:
        for s in servers['server_generation']:
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

            for d2 in datacenters['datacenter_id']:
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

            dismiss_count = int(dismiss[d, s].value() or 0)
            dismissed_servers = []
            for server_id, (server_gen, purchase_time) in list(current_fleet[d].items())[:dismiss_count]:
                if server_gen == s:
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

            for server_id, (server_gen, purchase_time) in current_fleet[d].items():
                if server_gen == s:
                    new_fleet[d][server_id] = (server_gen, purchase_time)

    utilization = utilized_capacity.value() / total_capacity.value() if total_capacity.value() != 0 else 0
    lifespan = lifespan_numerator.value() / total_servers.value() if total_servers.value() != 0 else 0

    logger.info(
        f"Time step {time_step} - Utilization: {utilization:.2f}, Lifespan: {lifespan:.2f}, Profit: {profit.value():.2f}")
    logger.info(f"Time step {time_step} - Total servers: {sum(len(servers) for servers in new_fleet.values())}")

    return actions, new_fleet, utilization, lifespan, profit.value()


# Function to optimize over multiple time steps, using incremental optimization
def solve_multi_time_steps(demand: pd.DataFrame,
                           datacenters: pd.DataFrame,
                           servers: pd.DataFrame,
                           selling_prices: pd.DataFrame,
                           total_time_steps: int = 168) -> List[Dict]:
    all_actions = []
    results = []
    current_fleet = {d: {} for d in datacenters['datacenter_id']}

    # Calculate the maximum purchase limit per time step
    max_purchase_per_step = calculate_max_purchase(demand, servers)
    logger.info(f"Calculated maximum purchase per step: {max_purchase_per_step}")

    # Lists to store values for each time step for plotting
    time_steps = []
    utilization_values = []
    lifespan_values = []
    profit_values = []

    for time_step in range(1, total_time_steps + 1):
        print(f"Solving for time step {time_step}")
        result = solve_fleet_optimization(
            demand, datacenters, servers, selling_prices, time_step, current_fleet, max_purchase_per_step
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

        # Store values for plotting
        time_steps.append(time_step)
        utilization_values.append(utilization)
        lifespan_values.append(lifespan)
        profit_values.append(profit)

        # Plotting the graphs
        plt.figure(figsize=(12, 8))

        # Utilization plot
        plt.subplot(3, 1, 1)
        plt.plot(time_steps, utilization_values, label='Utilization', color='b', marker='o')
        plt.xlabel('Time Step')
        plt.ylabel('Utilization')
        plt.title('Utilization Over Time Steps')
        plt.grid(True)

        # Lifespan plot
        plt.subplot(3, 1, 2)
        plt.plot(time_steps, lifespan_values, label='Lifespan', color='g', marker='o')
        plt.xlabel('Time Step')
        plt.ylabel('Lifespan')
        plt.title('Lifespan Over Time Steps')
        plt.grid(True)

        # Profit plot
        plt.subplot(3, 1, 3)
        plt.plot(time_steps, profit_values, label='Profit', color='r', marker='o')
        plt.xlabel('Time Step')
        plt.ylabel('Profit')
        plt.title('Profit Over Time Steps')
        plt.grid(True)

        plt.tight_layout()

        plt.show()
        plt.interactive(False)

    return all_actions


# Main function
def main():
    demand, datacenters, servers, selling_prices = load_problem_data()

    solution, results = solve_multi_time_steps(demand, datacenters, servers, selling_prices)

    if solution:
        save_solution(solution, "improved_solution.json")
        save_solution(results, "results.json")
        print("Solution saved to 'improved_solution.json'")
    else:
        print("Failed to find a solution")


if __name__ == "__main__":
    main()
