import torch
import os
import numpy as np
import gym
import utils
import time
import copy
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder


def evaluate(env, agent, video, num_episodes, L, step, test_env=False):
	episode_rewards = []
	for i in range(num_episodes):
		obs = env.reset()
		video.init(enabled=(i==0))
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(agent):
				action = agent.select_action(obs)
			obs, reward, done, _ = env.step(action)
			video.record(env)
			episode_reward += reward

		if L is not None:
			_test_env = '_test_env' if test_env else ''
			video.save(f'{step}{_test_env}.mp4')
			L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
		episode_rewards.append(episode_reward)
	
	return np.mean(episode_rewards)


def main(args):
	# Set seed
	utils.set_seed_everywhere(args.seed)

	# Initialize environments
	gym.logger.set_level(40)
	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='train'
	)
	test_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode=args.eval_mode,
		intensity=args.distracting_cs_intensity
	) if args.eval_mode is not None else None


	# Create working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, str(args.seed))
	print('Working directory:', work_dir)
	assert not os.path.exists(os.path.join(work_dir, 'train.log')), 'specified working directory already exists'
	utils.make_dir(work_dir)
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
	utils.write_info(args, os.path.join(work_dir, 'info.log'))

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	
	replay_buffer = utils.ReplayBuffer(
		obs_shape=env.observation_space.shape,
		action_shape=env.action_space.shape,
		capacity=args.train_steps//2,
		batch_size=args.batch_size,
		jump=args.jump
	)
	cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
	# print('Observations:', env.observation_space.shape)
	# print('Cropped observations:', cropped_obs_shape)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)

	if args.load_model is not None:
		print('load model')
		# saved_state = copy.deepcopy(agent)
		saved_agent = torch.load(os.path.join(args.load_model))
		saved_actor = saved_agent.actor.state_dict()
		saved_critic = saved_agent.critic.state_dict()
		saved_critic_target = saved_agent.critic_target.state_dict()
		# model_dict = agent.state_dict()

		agent_actor = agent.actor.state_dict()
		agent_critic = agent.critic.state_dict()
		agent_critic_target = agent.critic_target.state_dict()
		agent_actor.update(saved_actor)
		agent_critic.update(saved_critic)
		agent_critic_target.update(saved_critic_target)

		agent.actor.load_state_dict(agent_actor)
		agent.critic.load_state_dict(agent_critic)
		agent.critic_target.load_state_dict(agent_critic_target)

		agent.log_alpha = saved_agent.log_alpha
		
		


	start_step, episode, episode_reward, done = 0, 0, 0, True
	if args.algorithm == 'd2rl':
		L = Logger(work_dir,'d2rl')
	elif args.algorithm == 'sdac':
		L = Logger(work_dir,'sdac')
	elif args.algorithm == 'srp':
		L = Logger(work_dir,'srp')
	else:
		L = Logger(work_dir)
	start_time = time.time()
	for step in range(start_step, args.train_steps+1):
		if done:

			if step > start_step:
				L.log('train/duration', time.time() - start_time, step)
				start_time = time.time()
				L.dump(step)

			# # Evaluate agent periodically
			if step % args.eval_freq == 0 and step > 0:
				print('Evaluating:', work_dir)
				L.log('eval/episode', episode, step)
				# eval_flag = not eval_flag
				evaluate(env, agent, video, args.eval_episodes, L, step)
				if test_env is not None:
					evaluate(test_env, agent, video, args.eval_episodes, L, step, test_env=True)
				L.dump(step)

			# Save agent periodically
			if step > start_step and step % args.save_freq == 0:
				torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

			L.log('train/episode_reward', episode_reward, step)

			obs = env.reset()
			done = False
			episode_reward = 0
			episode_step = 0
			episode += 1


			L.log('train/episode', episode, step)

		# Sample action for data collection
		action = agent.sample_action(obs)

		# Run training update
		if step >= args.init_steps:
			num_updates = args.init_steps if step == args.init_steps else 1
			for _ in range(num_updates):
				agent.update(replay_buffer, L, step)

		# Take step
		next_obs, reward, done, _ = env.step(action)
		done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
		replay_buffer.add(obs, action, reward, next_obs, done_bool)
		episode_reward += reward
		obs = next_obs

		episode_step += 1

	print('Completed training for', work_dir)


if __name__ == '__main__':
	args = parse_args()
	main(args)
