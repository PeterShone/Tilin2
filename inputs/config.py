import os
from inputs.env import Environment
import math

################### DATA SET SELECTION ###################
# env_name = '30-60-90'
# env_name = '30-60-90+equilateral'
# env_name = '30-60-90+rectangle'
# env_name = "polyiamond-3"
# env_name = "45-45-90+rectangle"
# env_name = "labyrinth"
env_name = "L3+long_tile"

################### DATA SET CONFIGURATION ###################
# format: "tile set name : (mirror all tiles?, size of the superset, number of training data)"
env_attribute_dict = {
    '30-60-90'             : (True,  9, 12000),
    '30-60-90+equilateral' : (True,  15, 20),
    '30-60-90+rectangle'   : (True,  7, 12000),
    '45-45-90+rectangle'   : (False, 9,7000),
    "labyrinth"            : (False, 9, 5000),
    "polyiamond-3"         : (False, 9, 12000),
    "L3+long_tile"         : (False, 10, 2000),
}

symmetry_tiles, complete_graph_size, number_of_data = env_attribute_dict[env_name]
env_location = os.path.join('.', 'data', env_name)
environment = Environment(env_location, symmetry_tiles=symmetry_tiles)
# SET YOUR DATASET PATH HERE
# dataset_path = os.path.join('./dataset', f"{env_name}-ring{complete_graph_size}-{number_of_data}")
# dataset_path = '/research/dept8/fyp21/cwf2101/data/30-60-90+equilateral-ring15-2000'
dataset_path = '/research/dept8/fyp21/cwf2101/data/L3+long_tile-ring10-2000'

################### CREATING DATA ###################
shape_size_lower_bound=0.4
shape_size_upper_bound=0.6
max_vertices=20
validation_data_proportion=0.2

################### NETWORK PARAMETERS ###################
network_depth = 20
network_width = 32


################### TRAINING ###################
new_training = True
rand_seed = 2
batch_size = 1
learning_rate = 1e-3
training_epoch = 10000
save_model_per_epoch = 2
sample = False

COLLISION_WEIGHT    = 1/math.log(1+1e-1)
ALIGN_LENGTH_WEIGHT = 0.02
AVG_AREA_WEIGHT     = 1
SOL_WEIGHT     = 1.0


################### DEBUGGING ###################
debug_data_num = 5
debug_base_folder = ".."
experiment_id = 2000 # unique ID to identify an experiment

#################### TESTING ##################
output_tree_search_layout = False
silhouette_path = "/home/edwardhui/data/silhouette/selected_v2"
# network_path = f"./pre-trained_models/{env_name}.pth"
# network_path = f"./compare/No_sample/model_90_3.4976284503936768.pth"
# network_path = f"compare/sample-1/model_60_5.25051736831665.pth"
# network_path = f"compare/sample-10/model_24_21.027406692504883.pth"

### L shape ###
# network_path = f"pre-trained_models/model_18_L shape.pth"
# no sample
network_path = f"compare/L shape/No sample/model_42_3.5631096363067627.pth"
# network_path = f"compare/L shape/No sample/model_1_3.6349802017211914.pth"
# network_path = f"compare/L shape/No+/model_20_3.5711920261383057.pth"
# network_path = f"compare/L shape/No+/model_34_3.5643372535705566.pth"
# sample 1
# network_path = f"compare/L shape/sample-1/model_16_5.3590407371521.pth"
# network_path = f"compare/L shape/sample-1/model_30_5.359263896942139.pth"
# sample 10
# network_path = f"compare/L shape/sample-10/model_25_21.446828842163086.pth"
# network_path = f"compare/L shape/sample-10/model_3_21.704612731933594.pth"