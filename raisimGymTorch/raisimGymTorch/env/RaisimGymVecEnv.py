# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np
import platform
import os


class RaisimGymVecEnv:

    def __init__(self, impl, normalize_ob=True, seed=0, clip_obs=10.):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.normalize_ob = normalize_ob
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()
        self.num_critic_obs = self.wrapper.getCriticObDim()
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._critic_observation = np.zeros([self.num_envs, self.num_critic_obs], dtype=np.float32)
        self.actions = np.zeros([self.num_envs, self.num_acts], dtype=np.float32)
        self.log_prob = np.zeros(self.num_envs, dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.wrapper.setSeed(seed)
        self.count = 0.0
        self.mean = np.zeros(self.num_obs, dtype=np.float32)
        self.var = np.zeros(self.num_obs, dtype=np.float32)
        self.cmean = np.zeros(self.num_critic_obs, dtype=np.float32)
        self.cvar = np.zeros(self.num_critic_obs, dtype=np.float32)
        self._state = np.zeros([self.num_envs, 37], dtype=np.float32)
        self._ref = np.zeros([56, 37], dtype=np.float32)

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action):
        self.wrapper.step(action, self._reward, self._done)
        return self._reward.copy(), self._done.copy()

    def load_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.count = count
        self.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.var = np.loadtxt(var_file_name, dtype=np.float32)
        self.wrapper.setObStatistics(self.mean, self.var, self.count)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        self.wrapper.getObStatistics(self.mean, self.var, self.count)
        np.savetxt(mean_file_name, self.mean)
        np.savetxt(var_file_name, self.var)

    def observe(self, update_statistics=True):
        self.wrapper.observe(self._observation, update_statistics)
        return self._observation

    def load_critic_scaling(self, dir_name, iteration, count=1e5):
        cmean_file_name = dir_name + "/cmean" + str(iteration) + ".csv"
        cvar_file_name = dir_name + "/cvar" + str(iteration) + ".csv"
        self.count = count
        self.cmean = np.loadtxt(cmean_file_name, dtype=np.float32)
        self.cvar = np.loadtxt(cvar_file_name, dtype=np.float32)
        self.wrapper.setCriticObStatistics(self.cmean, self.cvar, self.count)

    def save_critic_scaling(self, dir_name, iteration):
        cmean_file_name = dir_name + "/cmean" + iteration + ".csv"
        cvar_file_name = dir_name + "/cvar" + iteration + ".csv"
        self.wrapper.getCriticObStatistics(self.cmean, self.cvar, self.count)
        np.savetxt(cmean_file_name, self.cmean)
        np.savetxt(cvar_file_name, self.cvar)

    def observe_critic(self, update_statistics=True):
        self.wrapper.observe_critic(self._critic_observation, update_statistics)
        return self._critic_observation

    def observe_state(self):
        self.wrapper.getState(self._state)
        return self._state

    def update_reference(self, ref, gait_num=-1):
        self._ref = ref
        self.wrapper.updateReference(self._ref, gait_num)

    def getVisEnvMode(self):
        return self.wrapper.getVisEnvMode()

    def get_reward_info(self):
        return self.wrapper.getRewardInfo()

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()
