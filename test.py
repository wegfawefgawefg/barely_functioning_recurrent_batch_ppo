import torch
import numpy as np

batch_size = 3
rollout_length = 4
num_recurrent_steps = 4
state_shape = 4

states = torch.rand((batch_size, rollout_length - 1))

a = torch.zeros([9]).view(-1, 3)
a[0][1] = 1
a[2][2] = 1
a[2][1] = 1
last_dones = a.nonzero()
print(last_dones)

print(states)

for batch_index, last_done in last_dones:
    print(f"{batch_index}, {last_done}")
    previous_episode_states = torch.arange(last_done+1)
    states[batch_index, previous_episode_states] = 0

print(states)
    