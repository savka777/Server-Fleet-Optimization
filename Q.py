
from copy import deepcopy
import pandas as pd
import numpy as np
import random
import uuid
import json
from scipy.stats import weibull_min

import evaluation

# Load the data
datacenters_df = pd.read_csv('data/datacenters.csv')
demand_df = pd.read_csv('data/demand.csv')
selling_prices_df = pd.read_csv('data/selling_prices.csv')
servers_df = pd.read_csv('data/servers.csv')

# Define the actions that will be taken at each state
actions = []
for server_type in servers_df['server_generation']:
    for dc_id in datacenters_df['datacenter_id']:
        actions.append(('buy', server_type, dc_id))  # Buy a server and deploy in a datacenter
        actions.append(('dismiss', server_type, dc_id))  # Dismiss a server in a datacenter
    
    for from_dc_id in datacenters_df['datacenter_id']:
        for to_dc_id in datacenters_df['datacenter_id']:
            if from_dc_id != to_dc_id:
                actions.append(('move', server_type, from_dc_id, to_dc_id))  # Move a server between datacenters

actions.append(('hold',))  # Do nothing - using a tuple here for consistency

# print(actions)

def convert_dict_to_tuple(d):
    """Recursively convert a dictionary to a tuple for hashing."""
    return tuple(sorted((k, convert_dict_to_tuple(v) if isinstance(v, dict) else v) for k, v in d.items()))

def hash_state(state):
    """Convert the state dictionary into a hashable tuple."""
    state_tuple = (
        state['time_step'],
        tuple(sorted(state['demand'].items())),
        tuple(sorted(state['required_servers'].items())),
        tuple(sorted(state['slot_requirements'].items())),
        tuple((dc_id, convert_dict_to_tuple(dc_data)) for dc_id, dc_data in sorted(state['datacenters'].items())),
        #round(state['profit'], 2)  # Round profit to 2 decimal places to reduce state space granularity
    )
    return state_tuple


# Check what the initial state at the moment is, initial state can be set to time_step 1 in demands
def initialize_state(time_step):
    demand_info = demand_df[demand_df['time_step'] == time_step]
    demand = {}
    required_servers = {}
    slot_requirements = {}

    for _, server in servers_df.iterrows():
        server_type = server['server_generation']
        server_capacity = server['capacity']
        
        demand[server_type] = demand_info[server_type].iloc[0] if server_type in demand_info.columns else 0
        required_servers[server_type] = demand[server_type] // server_capacity + (demand[server_type] % server_capacity > 0)
        slot_requirements[server_type] = server['slots_size']
    
    datacenters_state = {}
    for dc_id in datacenters_df['datacenter_id'].unique():
        slots_capacity = datacenters_df.loc[datacenters_df['datacenter_id'] == dc_id, 'slots_capacity'].values[0]
        datacenters_state[dc_id] = {
            'slots_capacity': slots_capacity,  # Initialize with total slots capacity
            'latency_sensitivity': datacenters_df.loc[datacenters_df['datacenter_id'] == dc_id, 'latency_sensitivity'].values[0],
            'allocated_servers': {server: 0 for server in servers_df['server_generation']},
            'operating_times': {server: 0.1 for server in servers_df['server_generation']}
        }
    
    state = {
        'time_step': time_step,
        'demand': demand,
        'required_servers': required_servers,
        'slot_requirements': slot_requirements,
        'datacenters': datacenters_state,
        # 'profit': 0
    }
    
    return state

# print("Initial State: ")
# print((initialize_state(1)))
# print('\n')


'''
This sections focuses on the OBjective 0 = U x L x P
'''


# def calculate_demand(state):
#     return state['demand']

# def calculate_capacity(state):
#     capacity = {}
#     for dc_id, datacenter in state['datacenters'].items():
#         for server_type, count in datacenter['allocated_servers'].items():
#             if count > 0:
#                 total_capacity = count * state['slot_requirements'][server_type]
#                 capacity[(server_type, datacenter['latency_sensitivity'])] = total_capacity
#     return capacity

# def calculate_utilization(demand, capacity):
#     utilization = {}
#     for key, cap in capacity.items():
#         server_type, latency_sensitivity = key
#         dem = demand.get(server_type, 0)
#         utilization[key] = min(cap, dem) / cap if cap > 0 else 0
#     return utilization

# def calculate_lifespan(state):
#     total_lifespan_ratio = 0
#     total_servers = 0
#     for dc_id, datacenter in state['datacenters'].items():
#         for server_type, operating_time in datacenter['operating_times'].items():
#             if operating_time > 0:
#                 life_expectancy = servers_df.loc[servers_df['server_generation'] == server_type, 'life_expectancy'].values[0]
#                 lifespan_ratio = operating_time / life_expectancy
#                 total_lifespan_ratio += lifespan_ratio
#                 total_servers += 1
#     return total_lifespan_ratio / total_servers if total_servers > 0 else 0

# def calculate_profit(demand, capacity, state):
#     revenue = 0
#     cost = 0
#     for key, cap in capacity.items():
#         server_type, latency_sensitivity = key
#         dem = demand.get(server_type, 0)
#         fulfilled_demand = min(cap, dem)
#         selling_price = selling_prices_df.loc[(selling_prices_df['server_generation'] == server_type) &
#                                               (selling_prices_df['latency_sensitivity'] == latency_sensitivity), 'selling_price'].values[0]
#         revenue += fulfilled_demand * selling_price
        
#         # Calculate cost for each server type in each datacenter
#         for dc_id, datacenter in state['datacenters'].items():
#             if datacenter['allocated_servers'][server_type] > 0:
#                 purchase_price = servers_df.loc[servers_df['server_generation'] == server_type, 'purchase_price'].values[0]
#                 maintenance_fee = servers_df.loc[servers_df['server_generation'] == server_type, 'average_maintenance_fee'].values[0]
#                 energy_cost = servers_df.loc[servers_df['server_generation'] == server_type, 'energy_consumption'].values[0]
                
#                 # Calculate costs proportionally to the number of servers
#                 num_servers = datacenter['allocated_servers'][server_type]
#                 operating_time = datacenter['operating_times'][server_type]
#                 total_maintenance_cost = maintenance_fee * operating_time * num_servers
#                 total_energy_cost = energy_cost * operating_time * num_servers
#                 total_purchase_cost = purchase_price * num_servers  # Only add this once, maybe during server purchase

#                 # Add all costs together
#                 cost += total_purchase_cost + total_maintenance_cost + total_energy_cost
                
#     return revenue - cost

# def calculate_objective(state):
#     demand = calculate_demand(state)
#     capacity = calculate_capacity(state)
#     utilization = calculate_utilization(demand, capacity)
#     lifespan = calculate_lifespan(state)
#     profit = calculate_profit(demand, capacity, state)
#     print(f"Demand: {demand}, Capacity: {capacity}")
#     print(f"Utilization: {utilization}")
#     print(f"Profit: {profit}")
    

    
#     # Calculate O as a product of utilization, lifespan, and profit
#     O = sum(utilization.values()) * lifespan * profit
#     print(f"Objective O: {O}")
#     return O

def reshape_demand_df(demand_df):
    return demand_df.melt(id_vars=['time_step', 'latency_sensitivity'], 
                          var_name='server_generation', 
                          value_name='demand')

def prepare_fleet_df(state, servers_df):
    fleet_list = []
    for dc_id, datacenter in state['datacenters'].items():
        for server_type, count in datacenter['allocated_servers'].items():
            if count > 0:
                for _ in range(count):
                    fleet_list.append({
                        'server_generation': server_type,
                        'datacenter_id': dc_id,
                        'latency_sensitivity': datacenter['latency_sensitivity'],
                        'capacity': servers_df.loc[servers_df['server_generation'] == server_type, 'capacity'].values[0],
                        'lifespan': datacenter['operating_times'][server_type],
                        'life_expectancy': servers_df.loc[servers_df['server_generation'] == server_type, 'life_expectancy'].values[0],
                        'purchase_price': servers_df.loc[servers_df['server_generation'] == server_type, 'purchase_price'].values[0],
                        'average_maintenance_fee': servers_df.loc[servers_df['server_generation'] == server_type, 'average_maintenance_fee'].values[0],
                         'cost_of_energy': datacenters_df.loc[datacenters_df['datacenter_id'] == dc_id, 'cost_of_energy'].values[0],
                        'energy_consumption': servers_df.loc[servers_df['server_generation'] == server_type, 'energy_consumption'].values[0],
                        'slots_size': servers_df.loc[servers_df['server_generation'] == server_type, 'slots_size'].values[0],
                        'cost_of_moving': servers_df.loc[servers_df['server_generation'] == server_type, 'cost_of_moving'].values[0],
                        'moved': 0  # Initialize moved status as 0 (not moved)
                        
                    })
    fleet_df = pd.DataFrame(fleet_list)
    return fleet_df

def calculate_demand_and_capacity(state, demand_df, servers_df):
    # Reshape the demand DataFrame
    demand_df_melted = reshape_demand_df(demand_df)
    
    # Extract the demand for the current time step
    time_step = state['time_step']
    D = evaluation.get_time_step_demand(demand_df_melted, time_step)

    if 'server_generation' not in D.columns:
        D = D.reset_index()  # This makes sure 'server_generation' becomes a column
    
    D_pivoted = D.pivot(index='server_generation', columns='latency_sensitivity', values='demand')
    
    # Prepare the fleet DataFrame from the state
    fleet_df = prepare_fleet_df(state, servers_df)
    
    # Calculate the capacity
    Z = evaluation.get_capacity_by_server_generation_latency_sensitivity(fleet_df)
    
    return D_pivoted, Z, fleet_df

def reshape_selling_prices(selling_prices_df):
    # Pivot the DataFrame so that latency_sensitivity becomes columns
    selling_prices_pivoted = selling_prices_df.pivot(index='server_generation', columns='latency_sensitivity', values='selling_price')
    
    return selling_prices_pivoted


def calculate_objective(state, demand_df, selling_prices_df, servers_df):
    # Prepare demand, capacity, and fleet DataFrame
    D, Z, fleet_df = calculate_demand_and_capacity(state, demand_df, servers_df)

    # Reshape selling_prices_df to the correct format
    selling_prices_df_pivoted = reshape_selling_prices(selling_prices_df)
    
    # Use the provided methods to calculate U, L, P
    U = evaluation.get_utilization(D, Z)
    L = evaluation.get_normalized_lifespan(fleet_df)
    P = evaluation.get_profit(D, Z, selling_prices_df_pivoted, fleet_df)
    
    # Calculate the overall objective O
    O = U * L * P
    return O

# Get the initial state of the datacenters
def calculate_initial_utilization(datacenter_id):
    capacity = datacenters_df.loc[datacenters_df['datacenter_id'] == datacenter_id, 'slots_capacity'].values[0]
    latency_sensitivity = datacenters_df.loc[datacenters_df['datacenter_id'] == datacenter_id, 'latency_sensitivity'].values[0]
    used_capacity = 0
    utilization = 0
    return capacity, latency_sensitivity, utilization, used_capacity

def buy_server(state, server_type, datacenter_id):
    # server_info = servers_df[servers_df['server_generation'] == server_type].iloc[0]
    server_slots = state['slot_requirements'][server_type]
    datacenter = state['datacenters'][datacenter_id]
    
    if datacenter['slots_capacity'] >= server_slots:
        datacenter['slots_capacity'] -= server_slots
        datacenter['allocated_servers'][server_type] += 1
        datacenter['operating_times'][server_type] += 1
        # Reduce the required servers
        server_capacity = servers_df.loc[servers_df['server_generation'] == server_type, 'capacity'].values[0]
        state['demand'][server_type] = max(0, state['demand'][server_type] - server_capacity)
        state['required_servers'][server_type] = max(0, state['required_servers'][server_type] - 1)

        # Calculate the overall objective O after the action
        # O = calculate_objective(state)
        O = calculate_objective(state, demand_df, selling_prices_df, servers_df)


        print(f"Purchased {server_type} at datacenter {datacenter_id}. Updated 0bj: {O} Remaining demand: {state['required_servers'][server_type]}")
    else:
        print(f"Cannot purchase {server_type} at datacenter {datacenter_id}: Not enough slots capacity.")

# print(calculate_initial_utilization("DC1"))
# print(calculate_initial_utilization("DC2"))
# print(calculate_initial_utilization("DC3"))
# print(calculate_initial_utilization("DC4"))

def move_server(state, server_type, from_datacenter_id, to_datacenter_id):
    # Find the number of slots we need to move 
    server_slots = state['slot_requirements'][server_type]
    # Get the datacenter we want to move the servers from
    from_datacenter = state['datacenters'][from_datacenter_id]
    # Get the datacenter we want to move the servers to
    to_datacenter = state['datacenters'][to_datacenter_id]
    
    # Make sure we have those servers
    if from_datacenter['allocated_servers'][server_type] > 0:
        # Make sure the destination datacenter has enough capacity
        if to_datacenter['slots_capacity'] >= server_slots:
            # Update, remove the servers from datacenter
            from_datacenter['allocated_servers'][server_type] -= 1
            # Update, add the amounts of slots avaliable
            from_datacenter['slots_capacity'] += server_slots
            
            # Update the new datacenter
            to_datacenter['allocated_servers'][server_type] += 1
            to_datacenter['slots_capacity'] -= server_slots
            
            # Transfer the operating time for the server
            to_datacenter['operating_times'][server_type] = from_datacenter['operating_times'][server_type]
               # Mark the server as moved in the fleet data
            for server in state['datacenters'][to_datacenter_id]['fleet']:
                if server['server_generation'] == server_type:
                    server['moved'] = 1  # Mark the server as moved

            # Calculate the overall objective O after the action
            # O = calculate_objective(state)
            O = calculate_objective(state, demand_df, selling_prices_df, servers_df)
            print(f"Moved {server_type} from {from_datacenter_id} to {to_datacenter_id}. Updated Obj: {O}")
        else:
            print(f"Cannot move server {server_type} from datacenter {from_datacenter_id} to {to_datacenter_id}: Not enough capacity.")
    else:
        print(f"Cannot move server {server_type}: No such server in datacenter {from_datacenter_id}.")


# When server retires replace if needed
def retire_and_replace_server(state, server_type, datacenter_id):
    datacenter = state['datacenters'][datacenter_id]
    server_info = servers_df[servers_df['server_generation'] == server_type].iloc[0]

    # Retire the server
    datacenter['allocated_servers'][server_type] -= 1
    # Give back the space to datacenter
    datacenter['slots_capacity'] += server_info['slots_size']

    # state['profit'] += server_info['purchase_price'] * 0.5  

    # Check if there is still demand for this server type
    if state['required_servers'][server_type] > 0:
        # Buy a replacement server
        buy_server(state, server_type, datacenter_id)
        print(f"Replaced server {server_type} in datacenter {datacenter_id}.")
    else:
        print(f"Retired server {server_type} in datacenter {datacenter_id} without replacement.")

# Update the operating times after each time step of all the servers active in each datacenter
# def update_operating_times(state):
#     # Update the operating times for each server type in each datacenter
#     for dc_id, datacenter in state['datacenters'].items():
#         for server_type, operating_time in datacenter['operating_times'].items():
#             if datacenter['allocated_servers'][server_type] > 0:
#                 datacenter['operating_times'][server_type] += 1

#                 # Check if the server has exceeded its lifespan 
#                 life_expectancy = servers_df.loc[servers_df['server_generation'] == server_type, 'life_expectancy'].values[0]
#                 if datacenter['operating_times'][server_type] >= life_expectancy:
#                     retire_and_replace_server(state, server_type, dc_id)



def dismiss_server(state, server_type, datacenter_id):
    #server_info = servers_df[servers_df['server_generation'] == server_type].iloc[0]
    #server_slots = state['slot_requirements'][server_type]
    datacenter = state['datacenters'][datacenter_id]
    
    if datacenter['allocated_servers'][server_type] > 0:
        # Calculate how much capacity is being lost
        server_capacity = servers_df.loc[servers_df['server_generation'] == server_type, 'capacity'].values[0]
        # Update the demand to reflect the loss of this server's capacity
        demand_increase = server_capacity  # This is the amount of demand that will be unmet

        # Re add this to the required servers count
        state['required_servers'][server_type] += 1 

        # Update the demand in the state
        state['demand'][server_type] += demand_increase

        # Free up the capacity in the datacenter
        datacenter['allocated_servers'][server_type] -= 1
        # datacenter['slots_capacity'] += server_slots
        datacenter['slots_capacity'] += servers_df.loc[servers_df['server_generation'] == server_type, 'slots_size'].values[0]
            
        # Reset operating time for the dismissed server
        datacenter['operating_times'][server_type] = 0.1
        # Calculate the overall objective O after the action
        # O = calculate_objective(state)
        O = calculate_objective(state, demand_df, selling_prices_df, servers_df)
        print(f"Dismissed {server_type} from {datacenter_id}. Updated Objective O: {O}")
    else:
        print(f"Cannot dismiss server {server_type}: No such server in datacenter {datacenter_id}.")

def hold(state):
    # Optionally calculate Objective O even if no action is taken
    # O = calculate_objective(state)
    O = calculate_objective(state, demand_df, selling_prices_df, servers_df)
    print(f"Held state. Objective O remains: {O}")


# After all functions are defined, you can add this test code
if __name__ == "__main__":

    # Initialize the state
    state = initialize_state(time_step=1)
    print("state: " ,state)
    print("\n")

    # Buy one server of type CPU.S1 in datacenter DC1
    buy_server(state, 'CPU.S1', 'DC1')
    hold(state)
    buy_server(state, 'CPU.S1', 'DC1')
    buy_server(state, 'CPU.S1', 'DC1')
    buy_server(state, 'CPU.S1', 'DC1')
    buy_server(state, 'CPU.S1', 'DC1')
    dismiss_server(state,'CPU.S1','DC1')
    buy_server(state, 'CPU.S1', 'DC1')

 
    # Check the state after the action
    print("\n")
    # print("State after buying CPU.S1 in DC1:")
    print(state['datacenters']['DC1'])

    # Calculate the objective
    O = calculate_objective(state, demand_df, selling_prices_df, servers_df)
    print("Objective O after action:", O)