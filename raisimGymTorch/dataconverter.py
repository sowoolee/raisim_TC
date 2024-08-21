import numpy as np
import os
import pickle as pkl

file_path = os.path.expanduser('~/Downloads/backflip.pkl')
with open(file_path, 'rb') as f:
    loaded_data = pkl.load(f)

state = np.concatenate([loaded_data['gc'][:,:,0:3], loaded_data['gc'][:,:,4:7], loaded_data['gc'][:,:,3:4],
                        loaded_data['gc'][:,:,7:], loaded_data['gv']], axis=-1)

# first_state = state[:,0:1,:]
last_state = state[:,-1:,:]
#
# first_chunk = np.repeat(first_state, 24, axis=1)
last_chunk = np.repeat(last_state, 174, axis=1) #68
#
state = np.concatenate([state, last_chunk], axis=1)
# state_extended = np.concatenate([first_chunk, state, last_chunk], axis=1)
#
# state_2d = state_extended.reshape(state_extended.shape[0], -1)

state_2d = state.reshape(-1, 37)
file_path = os.path.expanduser('~/Desktop/backflip__.csv')
np.savetxt(file_path, state_2d, delimiter=',')

print("here")