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

def train(num_timesteps, seed, load_path=None, itr = 1, simulate = False, render = False, vision=True):
	from baselines.common import set_global_seeds
	from baselines.common.vec_env.vec_normalize import VecNormalize
	from baselines.ppo2 import ppo2
	import tensorflow as tf
	from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

	def make_env():
		baseEnv = ReacherVisionEnv(xml_file='reacher/reacher_large_control.xml', random_init=True, vision=vision)
		# baseEnv = ReacherVisionEnv(random_color=True)
		env = bench.Monitor(baseEnv, logger.get_dir(), allow_early_resets=True, info_keywords = ("reward_dist", "reward_ctrl", "reach_dist"))
		return env

	env = DummyVecEnv([make_env])

	loadModel , loadEnv = None, None

	# if load_path!=None:
	# 	loadModel = load_path+"/%.5i" % itr
	# 	loadEnv = load_path+"/scaling"+str(itr)+".pkl"


	env = VecNormalize(env, loadFile = loadEnv)
	env.reset()
	if render:
		env.render()
		# for _ in range(100000):
		#     time.sleep(0.05)

	# ncpu = 1
	#
	# config = tf.ConfigProto(allow_soft_placement=True,
	# 						intra_op_parallelism_threads=ncpu,
	# 						inter_op_parallelism_threads=ncpu)
	#
	# config.gpu_options.allow_growth=True

	# with tf.Session(config=config) as sess:

	set_global_seeds(seed)
	# model = ppo2.learn(network='mlp', env=env, nsteps=2048, nminibatches=32,
	# model = ppo2.learn(network='mlp', env=env, nsteps=2048, nminibatches=64,
	# 				   # lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
	# 				   lam=0.97, gamma=0.99, noptepochs=10, log_interval=1,
	# 				   ent_coef=0.0,
	# 				   lr=3e-4,
	# 				   cliprange=0.2,
	# 				   total_timesteps=num_timesteps, save_interval=10,
	# 				   load_path=loadModel)
	if simulate:
		num_demos = 60000#200000
		cnt = 40000
		img_list = []
		act_list = []
		while cnt < num_demos:
			set_global_seeds(cnt)
			obs = np.zeros((env.num_envs,) + env.observation_space.shape)
			obs[:] = env.reset()

			rewardList = []
			imgs = []
			acts = []
			done = False
			while done==False:
				if render:
					# import pdb; pdb.set_trace()
					env.render()
					img = env.unwrapped.envs[0].env.get_current_image_obs()
					while (np.all(img == 0.)):
						env.render()
						img = env.unwrapped.envs[0].env.get_current_image_obs()
						time.sleep(0.05)
					imgs.append(img)
					time.sleep(0.05)

				# actions = model.step(obs)[0]
				actions = [np.random.uniform(low=env.unwrapped.envs[0].env.action_space.low, high=env.unwrapped.envs[0].env.action_space.high)]
				obs[:],_, dones,infos  = env.step(actions)

				reward = infos[0]['reward_dist']
				rewardList.append(reward)
				acts.append(actions[0])
				done = dones[0]
				if len(rewardList) == 1:#50:
					break
			img = env.unwrapped.envs[0].env.get_current_image_obs()
			imgs.append(img)
			print('Collecting demo %d' % cnt)
			img_list.append(np.array(imgs))
			act_list.append(np.array(acts))
			# import imageio
			# imageio.mimwrite('samp%d.gif' % num_gifs, imgs)
			cnt += 1
		print('Saving images')
		with open('/scr/kevin/unsupervised_upn/data/reacher/imgs_large_control20_20k.pkl', 'wb') as f:
			pickle.dump(np.array(img_list), f)
		print('Saving actions')
		with open('/scr/kevin/unsupervised_upn/data/reacher/acts_large_control20_20k.pkl', 'wb') as f:
			pickle.dump(np.array(act_list), f)

		return

def simulate(checkpointsDir, itr = 1, render = True, vision=True):
	return train(num_timesteps =  0, seed = 1, load_path=dirPREFIX+checkpointsDir, itr = itr,  simulate = True, render = render, vision=vision)


def run(args, expName):
	seed = args.seed
	vision = True if args.vision == 'True' else False

	expName = expName+'_seed'+str(seed)

	logger.configure(dir = dirPREFIX+expName)

	train(num_timesteps=N, seed = seed, vision=vision)

N = 1e7 ; simItr = 750
expName = 'reacher_visual_reward_bs_64_ctrl_eps_100_lam0.97'#_random_arm_color_lighting'#_handinit' #simItr=930 #1170 #1260 #1400 #1510 #4010
# expName = 'reacher_bs64_lam0.97'#_random_arm_color_lighting'#_handinit' #simItr=930 #1170 #1260 #1400 #1510 #4010
# mode = 'train'
mode = 'simulate'
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
