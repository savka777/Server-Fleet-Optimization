import uuid
from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from evaluation_v6 import get_actual_demand
from utils import load_problem_data, save_solution


def calculate_max_purchase(demand: pd.DataFrame, servers: pd.DataFrame) -> int:
    actual_demand = get_actual_demand(demand)

    print("Actual Demand DataFrame Structure:")
    print(actual_demand.head())
    print("\nColumns:", actual_demand.columns)
    print("\nData types:", actual_demand.dtypes)

    numeric_columns = actual_demand.select_dtypes(include=[np.number]).columns
    total_demand_per_timestep = actual_demand.groupby('time_step')[numeric_columns].sum().sum(axis=1)
    max_total_demand = total_demand_per_timestep.max()

    print(f"\nMaximum total demand: {max_total_demand}")

    min_server_capacity = servers['capacity'].min()
    print(f"Minimum server capacity: {min_server_capacity}")

    max_purchase = int(np.ceil(max_total_demand / min_server_capacity))
    print(f"Calculated max purchase: {max_purchase}")

    return max_purchase


def automatically_dismiss_servers(current_fleet: Dict[str, Dict[str, Tuple[str, int]]],
                                  servers: pd.DataFrame,
                                  time_step: int) -> Tuple[Dict[str, Dict[str, Tuple[str, int]]], List[Dict]]:
    dismissed_servers = []
    new_fleet = {d: {} for d in current_fleet}

    for datacenter, dc_servers in current_fleet.items():
        for server_id, (server_gen, purchase_time) in dc_servers.items():
            life_expectancy = servers[servers['server_generation'] == server_gen]['life_expectancy'].values[0]
            if time_step - purchase_time >= 96:  # Dismiss after 96 time steps
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


def solve_fleet_optimization_cp_sat(demand: pd.DataFrame,
                                    datacenters: pd.DataFrame,
                                    servers: pd.DataFrame,
                                    selling_prices: pd.DataFrame,
                                    time_step: int,
                                    current_fleet: Dict[str, Dict[str, str]],
                                    max_purchase_per_step: int) -> Tuple[
    List[Dict], Dict[str, Dict[str, str]], float, float, float]:
    current_fleet, auto_dismissed = automatically_dismiss_servers(current_fleet, servers, time_step)

    model = cp_model.CpModel()

    buy = {}
    move = {}
    dismiss = {}
    server_count = {}

    scaling_factor = 100

    for d in datacenters['datacenter_id']:
        for s in servers['server_generation']:
            buy[(d, s)] = model.NewIntVar(0, max_purchase_per_step * scaling_factor, f"buy_{d}_{s}")
            dismiss[(d, s)] = model.NewIntVar(0, current_fleet.get(d, {}).get(s, (0,))[0] * scaling_factor,
                                              f"dismiss_{d}_{s}")
            server_count[(d, s)] = model.NewIntVar(0, 1000 * scaling_factor,
                                                   f"server_count_{d}_{s}")
            for d2 in datacenters['datacenter_id']:
                if d != d2:
                    move[(d, d2, s)] = model.NewIntVar(0, current_fleet.get(d, {}).get(s, (0,))[0] * scaling_factor,
                                                       f"move_{d}_{d2}_{s}")

    for d in datacenters['datacenter_id']:
        model.Add(
            sum(server_count[(d, s)] * int(
                servers[servers['server_generation'] == s]['slots_size'].values[0] * scaling_factor)
                for s in servers['server_generation']) <=
            int(datacenters[datacenters['datacenter_id'] == d]['slots_capacity'].values[0] * scaling_factor)
        )

    for d in datacenters['datacenter_id']:
        for s in servers['server_generation']:
            current_count = sum(
                1 for server_gen in current_fleet.get(d, {}).values() if server_gen == s) * scaling_factor
            model.Add(
                server_count[(d, s)] == current_count + buy[(d, s)] +
                sum(move[(d2, d, s)] for d2 in datacenters['datacenter_id'] if d2 != d) -
                sum(move[(d, d2, s)] for d2 in datacenters['datacenter_id'] if d2 != d) -
                dismiss[(d, s)]
            )

    for s in servers['server_generation']:
        release_time = eval(servers[servers['server_generation'] == s]['release_time'].values[0])
        if time_step < release_time[0] or time_step > release_time[1]:
            for d in datacenters['datacenter_id']:
                model.Add(buy[(d, s)] == 0)

    # Define total_capacity as before
    total_capacity = sum(
        server_count[(d, s)] * int(
            servers[servers['server_generation'] == s]['capacity'].values[0] * scaling_factor)
        for d in datacenters['datacenter_id'] for s in servers['server_generation']
    )

    # Define total_demand based on the demand DataFrame
    numeric_columns = demand.select_dtypes(include=[np.number]).columns
    total_demand = int(demand[demand['time_step'] == time_step][numeric_columns].sum().sum() * scaling_factor)

    # Define utilized_capacity as a new decision variable
    utilized_capacity = model.NewIntVar(0, total_demand, "utilized_capacity")  # Set upper bound to total_demand

    # Add constraints to model utilized_capacity as the minimum of total_capacity and total_demand
    model.Add(utilized_capacity <= total_capacity)
    model.Add(utilized_capacity <= total_demand)

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Extract and print the value of utilized_capacity
        utilized_capacity_value = solver.Value(utilized_capacity)
        print(f"Utilized Capacity: {utilized_capacity_value}")

        # Similarly, extract other variables of interest like profit
        # Example: profit_value = solver.Value(profit)  # Assuming 'profit' is also a decision variable
        # print(f"Profit: {profit_value}")
    else:
        print("No feasible solution found.")

    total_servers = sum(
        server_count[(d, s)] for d in datacenters['datacenter_id'] for s in servers['server_generation'])

    lifespan_numerator = model.NewIntVar(0, int(1e9), "lifespan_numerator")
    model.Add(lifespan_numerator == sum(
        server_count[(d, s)] * time_step * scaling_factor
        for d in datacenters['datacenter_id'] for s in servers['server_generation']
    ))

    max_lifespan = sum(
        server_count[(d, s)] * int(
            servers[servers['server_generation'] == s]['life_expectancy'].values[0] * scaling_factor)
        for d in datacenters['datacenter_id'] for s in servers['server_generation']
    )

    model.Add(lifespan_numerator <= max_lifespan)

    # Calculate Revenue and Costs
    revenue = sum(
        utilized_capacity * int(selling_prices[(selling_prices['server_generation'] == s) &
                                               (selling_prices['latency_sensitivity'] ==
                                                datacenters[datacenters['datacenter_id'] == d][
                                                    'latency_sensitivity'].values[0])]['selling_price'].values[
                                    0] * scaling_factor)
        for d in datacenters['datacenter_id'] for s in servers['server_generation']
    )

    costs = sum(
        buy[(d, s)] * int(servers[servers['server_generation'] == s]['purchase_price'].values[0] * scaling_factor) +
        server_count[(d, s)] * (
                int(servers[servers['server_generation'] == s]['energy_consumption'].values[0] * scaling_factor) *
                int(datacenters[datacenters['datacenter_id'] == d]['cost_of_energy'].values[0] * scaling_factor) +
                int(servers[servers['server_generation'] == s]['average_maintenance_fee'].values[0] * scaling_factor)
        ) +
        sum(move[(d, d2, s)] * int(
            servers[servers['server_generation'] == s]['cost_of_moving'].values[0] * scaling_factor)
            for d2 in datacenters['datacenter_id'] if d2 != d)
        for d in datacenters['datacenter_id'] for s in servers['server_generation']
    )

    profit = model.NewIntVar(-int(1e9), int(1e9), "profit")
    model.Add(profit == revenue - costs)

    # Simplified objective function: maximize a weighted sum of utilization, lifespan, and profit
    model.Maximize(100 * utilized_capacity + 10 * lifespan_numerator + profit)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # After solving, extract the values properly
    total_capacity_value = solver.Value(total_capacity) / scaling_factor
    utilized_capacity_value = solver.Value(utilized_capacity) / scaling_factor

    utilization = utilized_capacity_value / total_capacity_value if total_capacity_value != 0 else 0

    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
        print(f"Problem not solved optimally for time step {time_step}. Status: {solver.StatusName(status)}")
        return None, current_fleet, 0, 0, 0

    actions = auto_dismissed.copy()
    new_fleet = {d: {} for d in datacenters['datacenter_id']}
    for d in datacenters['datacenter_id']:
        for s in servers['server_generation']:
            buy_count = solver.Value(buy[(d, s)]) // scaling_factor
            for _ in range(buy_count):
                server_id = str(uuid.uuid4())
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
                    move_count = solver.Value(move[(d, d2, s)]) // scaling_factor
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

            dismiss_count = solver.Value(dismiss[(d, s)]) // scaling_factor
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

    lifespan = solver.Value(lifespan_numerator) / solver.Value(max_lifespan) if solver.Value(max_lifespan) != 0 else 0

    print(
        f"Time step {time_step} - Utilization: {utilization:.2f}, Lifespan: {lifespan:.2f}, Profit: {solver.Value(profit):.2f}")

    return actions, new_fleet, utilization, lifespan, solver.Value(profit)


def solve_multi_time_steps_cp_sat(demand: pd.DataFrame,
                                  datacenters: pd.DataFrame,
                                  servers: pd.DataFrame,
                                  selling_prices: pd.DataFrame,
                                  total_time_steps: int = 168) -> List[Dict]:
    all_actions = []
    results = []
    current_fleet = {d: {} for d in datacenters['datacenter_id']}

    max_purchase_per_step = calculate_max_purchase(demand, servers)

    for time_step in range(1, total_time_steps + 1):

        print()
        print(f"Solving for time step {time_step}")

        result = solve_fleet_optimization_cp_sat(
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

    return all_actions, results


def main():
    demand, datacenters, servers, selling_prices = load_problem_data()

    solution, results = solve_multi_time_steps_cp_sat(demand, datacenters, servers, selling_prices)

    if solution:
        save_solution(solution, "improved_solution_cp_sat.json")
        save_solution(results, "results_cp_sat.json")
        print("Solution saved to 'improved_solution_cp_sat.json'")
        print("Results saved to 'results_cp_sat.json'")
    else:
        print("Failed to find a solution")


if __name__ == "__main__":
    main()
