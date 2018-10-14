from . import VecEnvWrapper
from baselines.common.running_mean_std import RunningMeanStd
import numpy as np
import pickle

class VecNormalize(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, loadFile = None):
        VecEnvWrapper.__init__(self, venv)
        if loadFile != None:
            self.loadScaling(loadFile)
        else:
            self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
            self.ret_rms = RunningMeanStd(shape=()) if ret else None
            self.clipob = clipob
            self.cliprew = cliprew
            self.ret = np.zeros(self.num_envs)
            self.gamma = gamma
            self.epsilon = epsilon

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """


        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)




        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        return self._obfilt(obs)

    def saveScaling(self, _file):
        info = {"ob_rms": self.ob_rms, "ret_rms": self.ret_rms, "clipob": self.clipob, "cliprew": self.cliprew, "ret": self.ret, "gamma": self.gamma, "epsilon": self.epsilon}
        fobj = open(_file, "wb")
        pickle.dump(info, fobj)
        fobj.close()

    def loadScaling(self, _file):
        fobj = open(_file, "rb")
        info = pickle.load(fobj)

        self.ob_rms = info["ob_rms"]
        self.ret_rms = info["ret_rms"]
        self.clipob = info["clipob"]
        self.cliprew = info["cliprew"]
        self.ret = info["ret"]
        self.gamma = info["gamma"]
        self.epsilon = info["epsilon"]
