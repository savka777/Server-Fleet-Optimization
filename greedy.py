import pandas as pd

# Import the data
server_df = pd.read_csv("data\servers.csv")
demand_df = pd.read_csv("data\demand.csv")
datacenters_df = pd.read_csv("data\datacenters.csv")

# Implementing a greedy approach
def greedy_allocation(time_step, server_df, demand_df, datacenters_df):
    actions = []
    # For each step identify the demand for each server type and categorize it by latency
    
    # Get the current step
    curr_demand_row = demand_df[demand_df['time_step'] == time_step]

    if not curr_demand_row.empty:
        # get the high/low demand
        high_demand = curr_demand_row[curr_demand_row['latency_sensitivity'] == 'high']
        low_demand = curr_demand_row[curr_demand_row['latency_sensitivity'] == 'low']

        # get the capicity of the data center
        datacenter_cap = datacenters_df.set_index('datacenter_id')['slots_capacity'].to_dict()

        # handle the high latency
        for _, server in server_df.iterrows():
            
            # check if server is within the necceary time step
            release = eval(server['release_time'])
            if release[0] <= time_step <= release[1]:
                # check the demand of the current server
                server_gen = server['server_generation']
                server_cap = server['capacity']
                demand = high_demand[server_gen].values[0]

                for dc_id, dc_latency in datacenters_df[['datacenter_id', 'latency_sensitivity']].values:
                    # make sure theres room in the datacenter to meet the demand (enough avaliable slots to accom the server were deploying)
                    if dc_latency == 'low' and datacenter_cap[dc_id] >= server_cap and demand > 0:
                        actions.append({
                            'action' : 'deploy',
                            'server-id' : server_gen,
                            'datacenters_id' : dc_id,
                            'latency sensitivity' : 'high',
                            'time_step' : time_step
                        })
                        # the demand thats been allocated to the datacenter
                        demand -= server_cap
                        # capicty filled at the datacenter
                        datacenter_cap[dc_id] -= server_cap
                        if demand <= 0 or datacenter_cap[dc_id] <=0:
                            break
                if demand <=0:
                    break
    return actions,datacenter_cap
        

# test 
actions, updated_datacenter_cap = greedy_allocation(1, server_df, demand_df, datacenters_df)

# create a DataFrame to display the remaining capacities

remaining_capacity_df = pd.DataFrame(list(updated_datacenter_cap.items()), columns=['Data Center ID', 'Remaining Capacity'])
remaining_capacity_df = remaining_capacity_df.merge(datacenters_df[['datacenter_id', 'slots_capacity', 'latency_sensitivity']], left_on='Data Center ID', right_on='datacenter_id')
remaining_capacity_df['Used Capacity'] = remaining_capacity_df['slots_capacity'] - remaining_capacity_df['Remaining Capacity']
remaining_capacity_df = remaining_capacity_df[['Data Center ID', 'latency_sensitivity', 'slots_capacity', 'Used Capacity', 'Remaining Capacity']]

print(remaining_capacity_df)


                    



                

               

                    



