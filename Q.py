from copy import deepcopy
import pandas as pd
import numpy as np
import random
import uuid
import json
from scipy.stats import weibull_min
import evaluation
import matplotlib.pyplot as plt

"""
Key Components:

1. Actions Definition:
   - A list of possible actions is defined, which includes buying, moving, and dismissing servers across different data centers, as well as holding the current state without any changes.

2. State Initialization (`initialize_state`):
   - This function initializes the state of the system for a given `time_step`.
   - The state includes current demand, required servers, slot requirements, and the configuration of each data center.

3. Fleet Management Functions:
   - `buy_server`: Purchases a server and deploys it to a specified data center, updating the state and recalculating the objective function.
   - `move_server`: Moves a server from one data center to another, updating the state and recalculating the objective function.
   - `dismiss_server`: Removes a server from a data center, updating the state, freeing up resources, and recalculating the objective function.
   - `hold`: Maintains the current state without any changes, and optionally recalculates the objective function.

4. Demand and Capacity Management:
   - `reshape_demand_df`: Reshapes the demand DataFrame to a long format where `server_generation` becomes a column. This is done to accomdate the evaluation functions given.
   - `prepare_fleet_df`: Prepares a DataFrame representing the current fleet of servers based on the state, including details like capacity, lifespan, and whether the server has been moved.
   - `calculate_demand_and_capacity`: Combines the demand and fleet data to calculate how well the current fleet meets the demand.

6. Objective Calculation (`calculate_objective`):
   - The objective function ( O ) is calculated as the product of utilization (U), lifespan (L), and profit (P).
   - This function is central to decision-making as it quantifies the effectiveness of fleet management actions.
   - The objective is recalculated after each action to assess its impact. (Import for later on using this as a reward mechanisim for the Q learning)

7. Example Execution:
   - Refer to main
   - It initializes the state for `time_step = 1`, performs a series of actions (buying, holding, dismissing servers), and prints the updated state and objective function after each action.

Future Steps:
- Implement constraint management*** to ensure that actions taken (buying or moving servers) do not violate limits such as slot capacity or energy consumption.
- Integrate Q-learning for reinforcement learning, enabling the system to learn and optimize decisions over time to maximize the long-term MAXIMUM objective function.
"""
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

def convert_to_hashable(data):
    if isinstance(data, dict):
        return tuple(sorted((k, convert_to_hashable(v)) for k, v in data.items()))
    elif isinstance(data, list):
        return tuple(convert_to_hashable(v) for v in data)
    else:
        return data

def hash_state(state):
    """Convert the state dictionary into a hashable tuple."""
    state_tuple = (
        state['time_step'],
        tuple(sorted(state['demand'].items())),
        tuple(sorted(state['required_servers'].items())),
        tuple(sorted(state['slot_requirements'].items())),
        tuple((dc_id, convert_to_hashable(dc_data)) for dc_id, dc_data in sorted(state['datacenters'].items())),
    )
    return state_tuple

# Check what the initial state at the moment is, initial state can be set to time_step 1 from demands df
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
            'operating_times': {server: 0.1 for server in servers_df['server_generation']} # operting time is set to 0.1 to avoid mulitplication by 0 in the first time_step
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
This sections focuses on the Objective 0 = U x L x P
'''

# Helper to reshape the deman df in order to comply with the parameters of the evaluation functions
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
                        'moved': 0
                        
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
      # Check if fleet_df is empty or malformed
    if fleet_df.empty:
        print("Warning: fleet_df is empty.")
        # Return empty DataFrame with the correct columns to avoid KeyError
        return D_pivoted, pd.DataFrame(columns=['server_generation', 'latency_sensitivity']), fleet_df
    
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

     # Check if the fleet DataFrame is empty or lacks necessary columns
    if fleet_df.empty or 'server_generation' not in fleet_df.columns or 'latency_sensitivity' not in fleet_df.columns:
        print("Fleet is empty or missing necessary columns, returning default objective value.")
        return 1e-6  # Return a very small default value to avoid division by zero or other issues

    # Reshape selling_prices_df to the correct format
    selling_prices_df_pivoted = reshape_selling_prices(selling_prices_df)
    
    # Use the provided methods to calculate U, L, P
    U = evaluation.get_utilization(D, Z)
    L = evaluation.get_normalized_lifespan(fleet_df)
    P = evaluation.get_profit(D, Z, selling_prices_df_pivoted, fleet_df)
    
    # Calculate the overall objective O
    O = U * L * P
    return O

# # Get the initial state of the datacenters
# def calculate_initial_utilization(datacenter_id):
#     capacity = datacenters_df.loc[datacenters_df['datacenter_id'] == datacenter_id, 'slots_capacity'].values[0]
#     latency_sensitivity = datacenters_df.loc[datacenters_df['datacenter_id'] == datacenter_id, 'latency_sensitivity'].values[0]
#     used_capacity = 0
#     utilization = 0
#     return capacity, latency_sensitivity, utilization, used_capacity

def calculate_reward(state, previous_O, action_result):
    """
    Calculate the reward for the current state based on the change in the objective function O
    and the result of the action.
    
    The reward is based on the improvement or degradation of the objective O after taking an action,
    and includes penalties for unsuccessful actions.
    """
    # Calculate the new objective after the action
    new_O = calculate_objective(state, demand_df, selling_prices_df, servers_df)
    
    # Reward is the difference between the new objective and the previous objective
    reward = new_O - previous_O
    
    # Penalize if the action failed due to constraints
    if not action_result['success']:
        reward -= action_result['penalty']  # Apply a penalty if the action failed

    return reward, new_O


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

        # Ensure the fleet key exists and append to it
        if 'fleet' not in datacenter:
            datacenter['fleet'] = []
        datacenter['fleet'].append({
            'server_generation': server_type,
            'datacenter_id': datacenter_id,
            'purchase_price': servers_df.loc[servers_df['server_generation'] == server_type, 'purchase_price'].values[0],
            'energy_consumption': servers_df.loc[servers_df['server_generation'] == server_type, 'energy_consumption'].values[0],
            'cost_of_moving': servers_df.loc[servers_df['server_generation'] == server_type, 'cost_of_moving'].values[0],
            'average_maintenance_fee': servers_df.loc[servers_df['server_generation'] == server_type, 'average_maintenance_fee'].values[0],
            'life_expectancy': servers_df.loc[servers_df['server_generation'] == server_type, 'life_expectancy'].values[0],
            'capacity': server_capacity,
            'slots_size': server_slots,
            'moved': 0,
        })

        # Calculate the overall objective O after the action
        # O = calculate_objective(state)
        O = calculate_objective(state, demand_df, selling_prices_df, servers_df)
        print(f"Purchased {server_type} at datacenter {datacenter_id}. Updated 0bj: {O} Remaining demand: {state['required_servers'][server_type]}")
        action_result = {
            'success': True,  # True if the action was successful, False otherwise
            'penalty': 0     # The penalty value
            }

    else:
        print(f"Cannot purchase {server_type} at datacenter {datacenter_id}: Not enough slots capacity.")
        action_result = {
            'success': False, 
            'penalty': -10     # The penalty value
            }
        
    return state, action_result

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
            #    # Mark the server as moved in the fleet data
            # for server in state['datacenters'][to_datacenter_id]['fleet']:
            #     if server['server_generation'] == server_type:
            #         server['moved'] = 1  # Mark the server as moved
             # Transfer the server from the fleet of the source datacenter to the destination datacenter
            server_to_move = None
            for server in from_datacenter['fleet']:
                if server['server_generation'] == server_type:
                    server_to_move = server
                    break
            
            if server_to_move:
                from_datacenter['fleet'].remove(server_to_move)
                server_to_move['datacenter_id'] = to_datacenter_id
                server_to_move['moved'] = 1
                if 'fleet' not in to_datacenter:
                    to_datacenter['fleet'] = []
                to_datacenter['fleet'].append(server_to_move)

            # Calculate the overall objective O after the action
            # O = calculate_objective(state)
            O = calculate_objective(state, demand_df, selling_prices_df, servers_df)
            print(f"Moved {server_type} from {from_datacenter_id} to {to_datacenter_id}. Updated Obj: {O}")
            action_results = {
                'success': True,  # True if the action was successful, False otherwise
                'penalty': 0     # The penalty value
                }
        else:
            print(f"Cannot move server {server_type} from datacenter {from_datacenter_id} to {to_datacenter_id}: Not enough capacity.")
            action_results = {
                'success': False,  # True if the action was successful, False otherwise
                'penalty': -10     # The penalty value
            }
    else:
        print(f"Cannot move server {server_type}: No such server in datacenter {from_datacenter_id}.")
        action_results = {
            'success': False,  # True if the action was successful, False otherwise
            'penalty': -10     # The penalty value
        }
    return state, action_results


# When server retires replace if needed (not sure if we need this yet)
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

'''This is done dynmamicaly now, leaving for now'''
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
         # Remove the server from the fleet
        for server in datacenter['fleet']:
            if server['server_generation'] == server_type:
                datacenter['fleet'].remove(server)
                break
        O = calculate_objective(state, demand_df, selling_prices_df, servers_df)
        print(f"Dismissed {server_type} from {datacenter_id}. Updated Objective O: {O}")
        action_results = {
            'success': True,  # True if the action was successful, False otherwise
            'penalty': 0     # The penalty value
        }
    else:
        print(f"Cannot dismiss server {server_type}: No such server in datacenter {datacenter_id}.")
        action_results = {
            'success': False,  # True if the action was successful, False otherwise
            'penalty': -10     # The penalty value
        }
    return state, action_results

def hold(state):
    # Optionally calculate Objective O even if no action is taken
    # O = calculate_objective(state)
    O = calculate_objective(state, demand_df, selling_prices_df, servers_df)
    print(f"Held state. Objective O remains: {O}")



# if __name__ == "__main__":

#     # Initialize the state
#     print("The Initial State: ")
#     state = initialize_state(time_step=1)
#     test = calculate_objective(state, demand_df, selling_prices_df, servers_df)
#     print(test)
#     print("state: " ,state)
#     print("\n")

#     buy_server(state, 'CPU.S1', 'DC1')
#     hold(state)
#     buy_server(state, 'CPU.S1', 'DC1')
#     buy_server(state, 'CPU.S1', 'DC1')
#     buy_server(state, 'CPU.S1', 'DC1')
#     buy_server(state, 'CPU.S1', 'DC1')
#     dismiss_server(state,'CPU.S1','DC1')
#     buy_server(state, 'CPU.S1', 'DC1')

 
#     # Check the state after the actions and Obj O, notice demand has decreased which is good
#     print("\n")
#     print(state["demand"])
#     O = calculate_objective(state, demand_df, selling_prices_df, servers_df)
#     print("Objective O after actions:", O)

# if __name__ == "__main__":
#     # Q-learning parameters
#     alpha = 0.1  # Learning rate
#     gamma = 0.9  # Discount factor
#     epsilon = 1.0  # Exploration rate
#     epsilon_decay = 0.995  # Decay rate for exploration
#     epsilon_min = 0.01  # Minimum exploration rate
#     num_episodes = 1000  # Number of episodes

#     # Initialize Q-table as a dictionary
#     Q_table = {}

#     def choose_action(state, actions, epsilon):
#         """Select an action using the epsilon-greedy strategy."""
#         if random.uniform(0, 1) < epsilon:
#             return random.choice(actions)  # Explore
#         else:
#             state_hash = hash_state(state)
#             if state_hash not in Q_table or len(Q_table[state_hash]) == 0:
#                 return random.choice(actions)  # Explore if state not in Q-table
#             else:
#                 return max(Q_table[state_hash], key=Q_table[state_hash].get)  # Exploit

#     def update_Q_table(state, action, reward, next_state):
#         """Update the Q-table using the Bellman equation."""
#         state_hash = hash_state(state)
#         next_state_hash = hash_state(next_state)

#         if state_hash not in Q_table:
#             Q_table[state_hash] = {act: 0 for act in actions}

#         best_next_action = max(Q_table.get(next_state_hash, {}), key=Q_table.get(next_state_hash, {}).get, default=0)
#         current_Q = Q_table[state_hash][action]

#         # Bellman equation
#         Q_table[state_hash][action] = current_Q + alpha * (reward + gamma * Q_table[next_state_hash].get(best_next_action, 0) - current_Q)

#     def perform_action(state, action):
#         """Perform the selected action and update the state."""
#         action_type = action[0]

#         if action_type == 'buy':
#             return buy_server(state, action[1], action[2])
#         elif action_type == 'move':
#             return move_server(state, action[1], action[2], action[3])
#         elif action_type == 'dismiss':
#             return dismiss_server(state, action[1], action[2])
#         elif action_type == 'hold':
#             hold(state)
#             return state, {'success': True, 'penalty': 0}

#     # Q-learning loop
#     for episode in range(num_episodes):
#         state = initialize_state(time_step=1)
#         previous_O = calculate_objective(state, demand_df, selling_prices_df, servers_df)
#         print(f"Initial Objective O: {previous_O}")

#         done = False
#         total_reward = 0

#         while not done:
#             time_step_actions = 0
#             while not done:
#                 action = choose_action(state, actions, epsilon)
#                 next_state, action_result = perform_action(state, action)
#                 reward, new_O = calculate_reward(state, previous_O, action_result)

#                 update_Q_table(state, action, reward, next_state)

#                 state = next_state
#                 previous_O = new_O
#                 total_reward += reward

#                 time_step_actions += 1

#                 # Check if demand is fully met or other stopping criteria
#                 if all(v == 0 for v in state['demand'].values()) or time_step_actions > 100:
#                     done = True
#                     break

#             # Simulate moving to the next time step
#             if not done:
#                 state = initialize_state(time_step=state['time_step'] + 1)

#         # Decay epsilon after each episode
#         if epsilon > epsilon_min:
#             epsilon *= epsilon_decay

#         print(f"Episode {episode + 1}, Objective O: {previous_O}, Total Reward: {total_reward}")

#     # After training, evaluate the policy by setting epsilon to 0 (pure exploitation)


import matplotlib.pyplot as plt
if __name__ == "__main__":
    # Q-learning parameters
    alpha = 0.2  # Learning rate
    gamma = 0.7  # Discount factor
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.7  # Decay rate for exploration
    epsilon_min = 0.01  # Minimum exploration rate
    num_episodes = 5  # Number of episodes

    # Initialize Q-table as a dictionary
    Q_table = {}
    
    # Lists to store results
    episode_rewards = []
    objective_O_values = []
    action_counts = []
    time_step_results = {}

    def choose_action(state, actions):
        """Select an action using the epsilon-greedy strategy."""
        if random.uniform(0, 1) < epsilon:
            return random.choice(actions)  # Explore
        else:
            state_hash = hash_state(state)
            if state_hash not in Q_table:
                return random.choice(actions)  # Explore if state not in Q-table
            else:
                return max(Q_table[state_hash], key=Q_table[state_hash].get)  # Exploit

    def update_Q_table(state, action, reward, next_state):
        """Update the Q-table using the Bellman equation."""
        state_hash = hash_state(state)
        next_state_hash = hash_state(next_state)

        if state_hash not in Q_table:
            Q_table[state_hash] = {act: 0 for act in actions}

        best_next_action = max(Q_table.get(next_state_hash, {}), key=Q_table.get(next_state_hash, {}).get, default=0)
        current_Q = Q_table[state_hash][action]

        # Bellman equation
        Q_table[state_hash][action] = current_Q + alpha * (reward + gamma * Q_table[next_state_hash].get(best_next_action, 0) - current_Q)

    def perform_action(state, action):
        """Perform the selected action and update the state."""
        action_type = action[0]

        if action_type == 'buy':
            return buy_server(state, action[1], action[2])
        elif action_type == 'move':
            return move_server(state, action[1], action[2], action[3])
        elif action_type == 'dismiss':
            return dismiss_server(state, action[1], action[2])
        elif action_type == 'hold':
            hold(state)
            return state, {'success': True, 'penalty': 0}
        else:
            return state, {'success': False, 'penalty': -10}

    # Q-learning loop
    for episode in range(num_episodes):
        
        state = initialize_state(time_step=1)
        previous_O = calculate_objective(state, demand_df, selling_prices_df, servers_df)
        print(f"Initial Objective O: {previous_O}")

        episode_reward = 0
        action_count = 0
        done = False
        time_step = 1

        while not done:
            time_step_actions = 0
            while not done:
                action = choose_action(state, actions)
                next_state, action_result = perform_action(state, action)
                reward, new_O = calculate_reward(state, previous_O, action_result)

                update_Q_table(state, action, reward, next_state)

                state = next_state
                previous_O = new_O

                episode_reward += reward
                action_count += 1

                time_step_actions += 1

                # Check if demand is fully met or other stopping criteria
                if all(v == 0 for v in state['demand'].values()) or time_step_actions > 500:
                    done = True
                    break

            # Store results by time step
            if time_step not in time_step_results:
                time_step_results[time_step] = {
                    'reward': 0,
                    'actions': 0,
                    'objective_O': 0
                }

            time_step_results[time_step]['reward'] += episode_reward
            time_step_results[time_step]['actions'] += action_count
            time_step_results[time_step]['objective_O'] += previous_O

            # Move to the next time step
            time_step += 1
            if time_step <= 168:
                state = initialize_state(time_step)
            else:
                done = True

        # Decay epsilon after each episode
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Store overall episode results
        episode_rewards.append(episode_reward)
        objective_O_values.append(previous_O)
        action_counts.append(action_count)

        print(f"Episode {episode + 1}, Objective O: {previous_O}, Reward: {episode_reward}, Actions: {action_count}")

    # After training, plot the results

    # Plot the rewards over episodes
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_episodes), episode_rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Episodes')
    plt.legend()
    plt.show()

    # Plot the objective function O over episodes
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_episodes), objective_O_values, label='Final Objective O per Episode', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Objective O')
    plt.title('Objective O over Episodes')
    plt.legend()
    plt.show()

    # Plot the action counts per episode
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_episodes), action_counts, label='Total Actions per Episode', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Total Actions')
    plt.title('Total Actions over Episodes')
    plt.legend()
    plt.show()

    # Plot results by time step
    time_steps = sorted(time_step_results.keys())
    rewards_by_time_step = [time_step_results[ts]['reward'] for ts in time_steps]
    actions_by_time_step = [time_step_results[ts]['actions'] for ts in time_steps]
    objective_O_by_time_step = [time_step_results[ts]['objective_O'] for ts in time_steps]

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, rewards_by_time_step, label='Total Reward per Time Step')
    plt.xlabel('Time Step')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Time Steps')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, objective_O_by_time_step, label='Objective O per Time Step', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Objective O')
    plt.title('Objective O over Time Steps')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, actions_by_time_step, label='Total Actions per Time Step', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('Total Actions')
    plt.title('Total Actions over Time Steps')
    plt.legend()
    plt.show()
