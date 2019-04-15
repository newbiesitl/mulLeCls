'''
Input data format
user_id, item_id, interaction_type
101, 110, 'click'
'''


import numpy as np


dat = [
    (101, 11, 'click'),
    (101, 10, 'click'),
    (101, 1, 'click'),
    (101, 2, 'click'),
    (111, 12, 'click'),
    (111, 11, 'click'),
    (111, 1, 'click'),
    (111, 10, 'click'),
]



# build item_id to matrix index mapping
item_id_set = set()
for each in dat:
    user_id, item_id, interaction_type = each
    if item_id in item_id_set:
        continue
    else:
        item_id_set.add(item_id)

# build item_id to index mapping
item_id_to_idx = dict(zip(list(item_id_set), range(len(item_id_set))))
idx_to_item_id = dict(zip(range(len(item_id_set)), list(item_id_set)))



total_items = len(list(item_id_to_idx.keys()))
utility_matrix = np.zeros((total_items, total_items))
print(utility_matrix.shape)
print(item_id_to_idx)

user_interactions = {}
for each in dat:
    user_id, item_id, interaction_type = each
    if user_id in user_interactions:
        user_interactions[user_id].append(item_id)
    else:
        user_interactions[user_id] = [item_id]




for user_id in user_interactions:
    user_history = user_interactions[user_id]
    for i in range(len(user_history)):
        for j in range(i, len(user_history)):
            this_item_id = user_history[i]
            other_item_id = user_history[j]
            this_item_idx = item_id_to_idx[this_item_id]
            other_item_idx = item_id_to_idx[other_item_id]
            utility_matrix[this_item_idx, other_item_idx] += 1

print(utility_matrix)


query_user_history = []