# Dataset parameters
sample_rate = 16000
hop_length = 384
max_length_sec = 10
classes = ['Speech',
           'Dog',
           'Cat',
           'Alarm_bell_ringing',
           'Dishes',
           'Frying',
           'Blender',
           'Running_water',
           'Vacuum_cleaner',
           'Electric_shaver_toothbrush']

threshold_range = [0.0, 1.0]
window_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
class_dependent = True

# Network parameters
state_size = 32

# VPG agent parameters
vpg_batch_size = 4
vpg_memory = 10000
vpg_learning_rate = 1e-3
vpg_update_frequency = vpg_batch_size
vpg_discount = 0.99
vpg_entropy_regularization = 1e-3
vpg_exploration = 0.5

# Training options
num_episodes = 1000000