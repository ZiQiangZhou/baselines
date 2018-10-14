#!/usr/bin/env python3
import numpy as np
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger

from multiworld.envs.mujoco.reacher.reacher import ReacherVisionEnv

import time
import sys
import pickle
import tensorflow as tf
import argparse

def query_yes_no(question, default="yes"):
	"""Ask a yes/no question via raw_input() and return their answer.

	"question" is a string that is presented to the user.
	"default" is the presumed answer if the user just hits <Enter>.
		It must be "yes" (the default), "no" or None (meaning
		an answer is required of the user).

	The "answer" return value is True for "yes" or False for "no".
	"""
	valid = {"yes": True, "y": True, "ye": True,
			 "no": False, "n": False}
	if default is None:
		prompt = " [y/n] "
	elif default == "yes":
		prompt = " [Y/n] "
	elif default == "no":
		prompt = " [y/N] "
	else:
		raise ValueError("invalid default answer: '%s'" % default)

	while True:
		sys.stdout.write(question + prompt)
		choice = input().lower()
		if default is not None and choice == '':
			return valid[default]
		elif choice in valid:
			return valid[choice]
		else:
			sys.stdout.write("Please respond with 'yes' or 'no' "
							 "(or 'y' or 'n').\n")

def train(num_timesteps, seed, load_path=None, itr = 1, simulate = False, render = False, vision=True):
	from baselines.common import set_global_seeds
	from baselines.common.vec_env.vec_normalize import VecNormalize
	from baselines.ppo2 import ppo2
	import tensorflow as tf
	from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

	def make_env():
		baseEnv = ReacherVisionEnv(vision=vision)
		# baseEnv = ReacherVisionEnv(random_color=True)
		env = bench.Monitor(baseEnv, logger.get_dir(), allow_early_resets=True, info_keywords = ("reward_dist", "reward_ctrl", "reach_dist"))
		return env

	env = DummyVecEnv([make_env])

	loadModel , loadEnv = None, None

	if load_path!=None:
		loadModel = load_path+"/%.5i" % itr
		loadEnv = load_path+"/scaling"+str(itr)+".pkl"


	env = VecNormalize(env, loadFile = loadEnv)
	env.reset()
	if render:
		env.render()
		# for _ in range(100000):
		#     time.sleep(0.05)

	ncpu = 1

	config = tf.ConfigProto(allow_soft_placement=True,
							intra_op_parallelism_threads=ncpu,
							inter_op_parallelism_threads=ncpu)

	config.gpu_options.allow_growth=True

	with tf.Session(config=config) as sess:

		set_global_seeds(seed)
		# model = ppo2.learn(network='mlp', env=env, nsteps=2048, nminibatches=32,
		model = ppo2.learn(network='mlp', env=env, nsteps=2048, nminibatches=64,
						   # lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
						   lam=0.97, gamma=0.99, noptepochs=10, log_interval=1,
						   ent_coef=0.0,
						   lr=3e-4,
						   cliprange=0.2,
						   total_timesteps=num_timesteps, save_interval=10,
						   load_path=loadModel)
		if simulate:
			num_gifs = 0
			while True:
				obs = np.zeros((env.num_envs,) + env.observation_space.shape)
				obs[:] = env.reset()

				rewardList = []
				imgs = []
				done = False
				while done==False:
					if render:
						# import pdb; pdb.set_trace()
						env.render()
						img = env.unwrapped.envs[0].env.get_current_image_obs()
						imgs.append(img)
						time.sleep(0.05)

					actions = model.step(obs)[0]
					obs[:],_, dones,infos  = env.step(actions)


					reward = infos[0]['reward_dist']
					print(reward)
					print('reach reward is', infos[0]['reward_dist'])
					print('reach dist is', infos[0]['reach_dist'])
					# print('distractor position is', env.unwrapped.envs[0].env.wrapped_env.get_distr_pos())

					rewardList.append(reward)
					done = dones[0]
					if len(rewardList) == 100:#50:
						break
				img = env.unwrapped.envs[0].env.get_current_image_obs()
				imgs.append(img)
				import imageio
				imageio.mimwrite('samp%d.gif' % num_gifs, imgs)
				num_gifs += 1
				if not query_yes_no('Continue simulation?'):
					break

			return rewardList

def simulate(checkpointsDir, itr = 1, render = True, vision=True):
	return train(num_timesteps =  0, seed = 1, load_path=dirPREFIX+checkpointsDir, itr = itr,  simulate = True, render = render, vision=vision)


def run(args, expName):
	seed = args.seed
	vision = True if args.vision == 'True' else False

	expName = expName+'_seed'+str(seed)

	logger.configure(dir = dirPREFIX+expName)

	train(num_timesteps=N, seed = seed, vision=vision)

N = 1e7 ; simItr = 750
expName = 'reacher_visual_reward_bs_64_ctrl_eps_100_lam0.97_img100'#_random_arm_color_lighting'#_handinit' #simItr=930 #1170 #1260 #1400 #1510 #4010
# expName = 'reacher_bs64_lam0.97'#_random_arm_color_lighting'#_handinit' #simItr=930 #1170 #1260 #1400 #1510 #4010
mode = 'train'
# mode = 'simulate'
parser = argparse.ArgumentParser()
parser.add_argument('--vision', type=str, default='True')

if mode == 'train':
	dirPREFIX = '/scr/kevin/baselines/data/'
	parser.add_argument('--seed', type=int)

	args = parser.parse_args()
	assert args.seed!=None
	run(args, expName)
else:
	assert mode =='simulate'

	seed = 0
	parser.add_argument('--simItr', type=int)
	args = parser.parse_args()

	dirPREFIX = '/scr/kevin/baselines/data/'#Sawyer-PickPlace/ppoContextual/'
	simulate(expName+'_seed'+str(seed)+'/checkpoints', itr = args.simItr)
