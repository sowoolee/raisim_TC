import numpy as np
import os
import pickle as pkl

import raisimpy as raisim

# world = raisim.World()
# ground = world.addGround()
# anymal = world.addArticulatedSystem(os.path.dirname(os.path.abspath(__file__)) + "/../rsc/go1/go1.urdf")
#
# server = raisim.RaisimServer(world)
#
# server.launchServer(8081)
# anymal = server.addVisualArticulatedSystem("go1", os.path.dirname(os.path.abspath(__file__)) + "/../rsc/go1/go1.urdf")
# anymal.setGeneralizedCoordinate(np.array([0, 0, 3.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8]))

file_path = os.path.expanduser('~/Downloads/backflip.pkl')
with open(file_path, 'rb') as f:
    loaded_data = pkl.load(f)

state = np.concatenate([loaded_data['gc'][:,:,0:3], loaded_data['gc'][:,:,4:7], loaded_data['gc'][:,:,3:4],
                        loaded_data['gc'][:,:,7:], loaded_data['gv']], axis=-1)

# first_state = state[:,0:1,:]
last_state = state[:,-1:,:]

last_quaternion = state[:,-1,6]
indices = np.where(last_quaternion >= -0.9)[0]
values = last_quaternion[indices]

state = np.delete(state, indices, axis=0)
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