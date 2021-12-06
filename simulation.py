#%% VS Code Notebook
import copy
import functools
import json
import geopandas as gpd
import numpy as np
import pandas as pd
import random
import sys

from itertools import product
from pathlib import Path
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm


root = Path(__file__).parent

df_heat = pd.read_csv(root/'data/san_francisco-traversals.csv.gz')
df_time = pd.read_csv(root/'data/san_francisco-censustracts-2020-1-WeeklyAggregate.csv.gz')
with open(root/'data/san_francisco_censustracts.json') as f:
    geo_data = json.load(f)
geo_rows = []
for geo_region in geo_data['features']:
    for poly in [Polygon(y) for z in geo_region['geometry']['coordinates'] for y in z]:
        geo_rows.append({
            'poly': poly,
            'movement_id': geo_region['properties']['MOVEMENT_ID'],
            'display_name': geo_region['properties']['DISPLAY_NAME'],
        })
df_geo = gpd.GeoDataFrame(geo_rows)
del geo_rows
del geo_data
# %%
df_heat.head()
# %%
df_time.head()
# %%
df_geo.head()

# %%
weekend_probability = 2/7
# TODO: I think this is Shapely serialized
to_cooranate_pairs = lambda x: np.array([[float(z) for z in y.split(' ')] for y in x[10:-2].split(', ')])
get_mean_point = lambda x: np.mean(x, axis=0)
sampling_weekend_df = df_heat[df_heat['dayType']=='weekend']
sampling_weekday_df = df_heat[df_heat['dayType']=='weekday']
def sample_start_end_points(n,day_type=None):
    if day_type is None:
        if random.random() > weekend_probability:
            day_type='weekday'
        else:
            day_type='weekend'

    if day_type == 'weekday':
        sampling_df = sampling_weekday_df    
    else:
        sampling_df = sampling_weekend_df
    
    sample_rows = sampling_df['wktGeometry'].sample(n,weights = sampling_df.traversals)
    return [get_mean_point(to_cooranate_pairs(x)) for x in sample_rows]

sampled_points = sample_start_end_points(2,'weekend')
start = sampled_points[0]
end = sampled_points[1]
print(start)
print(end)
# %%
geo_series = gpd.GeoSeries(df_geo.poly)
@functools.lru_cache(maxsize=100000) # Get better performance with low effort
def get_movement_id_helper(point):
    selected = df_geo['movement_id'][geo_series.contains(Point(point))]
    if len(selected) == 0:
        raise ValueError(f'No movement id found for point {point}')
    return int(selected.iloc[0])

def get_movement_id(point):
    return get_movement_id_helper(tuple(point))

get_movement_id(start)
# %%
@functools.lru_cache(maxsize=100000) # Get better performance with low effort
def get_time_between_points_helper(start, end):
    selected = df_time['mean_travel_time'][
        (df_time['sourceid']==get_movement_id(start)) & 
        (df_time['dstid']==get_movement_id(end))
    ]
    if len(selected) == 0:
        raise ValueError(f'Travel time data could not be found between points')
    return float(selected.iloc[0])

# Need to do some casting here to make things hashable
def get_time_between_points(start, end):
    return get_time_between_points_helper(
        tuple(start.tolist()),
        tuple(end.tolist())
    )

get_time_between_points(start, end)
# %%
start = sample_start_end_points(1,'weekend')[0]
end = sample_start_end_points(1,'weekend')[0]
get_time_between_points(start, end)
# %%
class Round:
    def __init__(self,n,k,day_type=None):
        # Setting:
        self.n = n
        self.k = k
        sampled_points = sample_start_end_points(n+1,day_type)
        self.start_point = sampled_points.pop()
        self.end_points = sampled_points
        
        # State:
        self.next_index = 0
        self.cost = 0
        self.benefit = 0
        self.accepted = []
        self.rejected = []

    def reset(self):
        self.next_index = 0
        self.cost = 0
        self.benefit = 0
        self.accepted = []
        self.rejected = []
        return self

    @property
    def reward(self):
        return self.benefit - self.cost

    def accept(self,point):
        self.cost += self.c(point)
        self.benefit += self.f(point)
        self.accepted.append(point)
        self.next_index += 1

    def reject(self,point):
        self.accepted.append(point)
        self.next_index += 1

    def get_additional_time(self,point):
        return get_time_between_points(point, self.accepted[-1] if len(self.accepted) > 0 else self.start_point)

    # Cost
    def c(self,point):
        # Based on https://www.sfmta.com/sites/default/files/reports-and-documents/2019/01/mobility_report_accessible_final.pdf
        cost_from_distance = 15.25*0.56/60/60
        # Based on https://www.dir.ca.gov/dlse/faq_minimumwage.htm
        cost_from_time = 14.00/60/60
        cost_per_second = cost_from_time*cost_from_distance
        return self.get_additional_time(point)*cost_per_second 

    # TODO: Update based on original location
    # Benefit
    def f(self,point):
        # Based on https://www.uber.com/global/en/price-estimate/
        discount_factor = 0.8 ** (len(self.accepted))
        # Based on https://www.uber.com/global/en/price-estimate/
        revenue_from_time = 0.39/60
        # Based on https://www.sfmta.com/sites/default/files/reports-and-documents/2019/01/mobility_report_accessible_final.pdf
        revenue_from_distance = 15.25*0.91/60/60 # Converted to time
        # Based on https://www.uber.com/gh/en/drive/basics/tracking-your-earnings/
        uber_s_cut_discount = 0.75
        revenue_per_second = (revenue_from_time+revenue_from_distance) * uber_s_cut_discount
        return get_time_between_points(point, self.start_point)*revenue_per_second*discount_factor

    def g(self,point):
        return self.f(point)-self.c(point)

    def stream(self):
        for end_point in self.end_points:
            if len(self.accepted) < self.k:
                yield end_point
            else:
                break
Round(5,3)
# %%
# Input: Stream of elements V , scaled objective g˜ = f − 2c.
# Output: Solution Q.
# 1: Q ← ∅
# 2: for each arriving element e do
# 3:     if g˜(e|Q) > 0 then
# 4:         Q ← Q ∪ {e}
# 5:     end if
# 6: end for
# 7: return Q

def Online_CSG(round:Round):
    for e in round.stream():
        if round.g(e) > 0:
            round.accept(e)
    return round

Online_CSG(Round(5,3))
# %%
# Input: Stream of elements V , scaled objective g˜ = f − s · c (s ≥ 1 is an absolute constant), cardinality
# k, threshold τ .
# Output: Solution Q.
# 1: Q ← ∅
# 2: while stream not empty do
#     3: e ←next stream element
#     4: if g˜(e|Q) ≥ τ and |Q| < k then
#         5: Q ← Q ∪ {e}
#     6: end if
# 7: end while
# 8: return Q

def Streaming_CSG(round:Round,tau:float):
    for e in round.stream():
        if round.g(e) >= tau:
            round.accept(e)
    return round

#%%
def Brute_Force_Optimal(round:Round):
    cartesian_product = product([0,1], repeat=round.n)
    best_reward = 0
    best_round = []
    for solution in cartesian_product:
        if sum(solution) <= round.k:
            round.reset()
            all_accepted = [x for (x,y) in zip(round.end_points, solution) if y == 1]
            for point in all_accepted:
                round.accept(point)
            if round.reward > best_reward:
                best_reward = round.reward
                best_round = copy.deepcopy(round)
    return best_round

#%%
# o_list = []
# s_list = []
# b_list = []
# for i in tqdm(range(500)):
#     while True:
#         round = Round(1,3)
#         try:
#             b_list.append(Brute_Force_Optimal(round)[1])
#             round.reset()
#             s_list.append(Streaming_CSG(round)[1])
#             round.reset()
#             o_list.append(Online_CSG(round)[1])
#         except ValueError:
#             continue # Missing data for given points, resample
#         break
# print(np.mean(o_list))
# print(np.mean(s_list))
# print(np.mean(b_list))
# %%
n = 4
k = 2
get_round = lambda: Round(n,k)

def get_theoretical_tau(get_round):
    cost_list = []
    benefit_list = []
    rounds = []
    for _ in tqdm(range(1000)):
        while True:
            try:
                round = get_round()
                optimal_round = Brute_Force_Optimal(round)
                benefit_list.append(optimal_round.benefit)
                cost_list.append(optimal_round.cost)
                rounds.append(round)
            except ValueError:
                continue
            break

    q = 0.381966011 # 1/2(3-sqrt(5)
    theoretical_tau = (1/k)*(q*np.mean(benefit_list) - np.mean(cost_list))
    return theoretical_tau, rounds
# theoretical_tau,good_rounds = get_theoretical_tau(get_round)
# %%
def get_total_reward(rounds:list,algorithm):
    return sum(algorithm(round.reset()).reward for round in rounds)

def train_tau(rounds,tau_init=10,tau_step=0.1):
    tau = tau_init
    last_reward = get_total_reward(rounds,lambda x: Streaming_CSG(x,tau=tau))
    direction = None
    with tqdm() as pbar:
        while True:
            if direction == 'up' or direction is None:
                reward = get_total_reward(rounds,lambda x: Streaming_CSG(x,tau=tau+tau_step))
                if reward > last_reward:
                    tau += tau_step
                    direction = 'up'
                elif direction is None:
                    direction = 'down'
                else:
                    break
            else:
                reward = get_total_reward(rounds,lambda x: Streaming_CSG(x,tau=tau-tau_step))
                if reward > last_reward:
                    tau -= tau_step
                else:
                    break
            last_reward = reward
            pbar.update(1)
        
    return tau

# good_rounds = [Round(4,3) for _ in range(50)]
# train_tau(good_rounds[:500])
# %%
def get_results(get_round):
    print('Getting theoretical tau value:')
    theoretical_tau,good_rounds = get_theoretical_tau(get_round)
    print('Getting trained tau value:')
    trained_tau = train_tau(good_rounds)

    print('Getting algorithm performance:')
    b_list = []
    o_list = []
    s_trained_list = []
    s_theoretical_list = []
    for i in tqdm(range(1000)):
        while True:
            round = get_round()
            try:
                b_list.append(Brute_Force_Optimal(round).reward)
                round.reset()
                s_trained_list.append(Streaming_CSG(round,tau=trained_tau).reward)
                round.reset()
                s_theoretical_list.append(Streaming_CSG(round,tau=theoretical_tau).reward)
                round.reset()
                o_list.append(Online_CSG(round).reward)
            except ValueError:
                continue # Missing data for given points, resample
            break
    return {
        'brute_force_data': b_list,
        'online_data': o_list,
        'streaming_trained_data': s_trained_list,
        'streaming_theoretical_data': s_theoretical_list,
        'trained_tau': trained_tau,
        'theoretical_tau': theoretical_tau,
    }

args = sys.argv[1:]
k = int(args[0])
# n = 2
for n in range(1,11):
    print(f'n = {n}')
    # k = 5
    get_round = lambda: Round(n,k)

    dir_path = Path('./batch2')
    dir_path.mkdir(parents=True, exist_ok=True)
    output_path = dir_path/f'n-{n}-k-{k}.json'

    with open(output_path,'w') as result_file:
        json.dump(get_results(get_round),result_file)
# %%
