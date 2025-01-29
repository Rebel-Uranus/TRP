import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import itertools

import utils
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC


# Disentangled Deep Reinforcement Learning
class D2RL(SAC):
	def __init__(self, obs_shape, action_shape, args):
		self.aux_update_freq = args.aux_update_freq
		self.soda_batch_size = args.soda_batch_size
		self.soda_tau = args.soda_tau
		self.mi_loss = args.mi_loss
		self.classify_loss = args.classify_loss
		self.kl_loss = args.kl_loss
		print(self.kl_loss, self.mi_loss, self.classify_loss)

		self.discount = args.discount
		self.critic_tau = args.critic_tau
		self.encoder_tau = args.encoder_tau
		self.actor_update_freq = args.actor_update_freq
		self.critic_target_update_freq = args.critic_target_update_freq

		self.shared_cnn = m.SharedCNN(obs_shape, args.num_shared_layers, args.num_filters).cuda()
		print(self.shared_cnn.out_shape)
		self.head_cnn = m.D2RL_HeadCNN(self.shared_cnn.out_shape, args.num_head_layers, args.num_filters, args.embed_dim).cuda()
		print(2)
		actor_encoder = m.Encoder(
			self.shared_cnn,
			self.head_cnn,
			m.D2RL_RLProjection(self.head_cnn.out_shape, args.projection_dim)
		)
		print(3)
		critic_encoder = m.Encoder(
			self.shared_cnn,
			self.head_cnn,
			m.D2RL_RLProjection(self.head_cnn.out_shape, args.projection_dim)
		)
		print('encode_net build')
		self.actor = m.Actor(actor_encoder, action_shape, args.hidden_dim, args.actor_log_std_min, args.actor_log_std_max).cuda()
		self.critic = m.Critic(critic_encoder, action_shape, args.hidden_dim).cuda()
		self.critic_target = deepcopy(self.critic)
		print('actor & critic build')
		# self.augment_classifier = NonlinearOrderClassifier(emb_size = self.head_cnn.out_shape[0]).cuda()
		# print('classifier_net build')
		self.mi_estimator = m.CLUBSample(self.head_cnn.out_shape[0], self.head_cnn.out_shape[0], self.head_cnn.out_shape[0]).cuda()
		self.mi_estimator_sigma = m.CLUBSample(self.head_cnn.out_shape[0], self.head_cnn.out_shape[0], self.head_cnn.out_shape[0]).cuda()

		# new_dim = int(self.head_cnn.out_shape[0]/2)
		# self.mu_prediction = nn.Sequential(
		# 	nn.Linear(self.head_cnn.out_shape[0], new_dim), 
 		# 	nn.ReLU(), 
 		# 	nn.Linear(new_dim, self.head_cnn.out_shape[0])
		# ).cuda()

		# self.var_prediction = nn.Sequential(
		# 	nn.Linear(self.head_cnn.out_shape[0], new_dim), 
 		# 	nn.ReLU(), 
 		# 	nn.Linear(new_dim, self.head_cnn.out_shape[0]),
		# 	nn.Tanh() 
		# ).cuda()

		print('emb_projection_net build')

		self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda()
		self.log_alpha.requires_grad = True
		self.target_entropy = -np.prod(action_shape)

		self.actor_optimizer = torch.optim.Adam(
			self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
		)
		self.critic_optimizer = torch.optim.Adam(
			self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
		)
		self.log_alpha_optimizer = torch.optim.Adam(
			[self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
		)
		# self.d2rl_classify_optimizer = torch.optim.Adam(
		# 	self.augment_classifier.parameters(), lr=args.aux_lr, betas=(args.aux_beta, 0.999)
		# )
		# sampler_optimizer = torch.optim.Adam(sampler.parameters(), lr = lr)
		self.mi_optimizer = torch.optim.Adam(self.mi_estimator.parameters(), lr = 1e-4)
		self.mi_optimizer_sigma = torch.optim.Adam(self.mi_estimator_sigma.parameters(), lr = 1e-4)
		self.encoder_optimizer = torch.optim.Adam(
			itertools.chain(self.shared_cnn.parameters(),
			 				self.head_cnn.parameters()#,
			  				# self.augment_classifier.parameters()
							),
			lr=args.aux_lr, 
			betas=(args.aux_beta, 0.999)
		)
		# self.encoder_optimizer = torch.optim.Adam(
		# 	itertools.chain(self.shared_cnn.parameters(), self.head_cnn.parameters(), self.mu_prediction.parameters(), self.var_prediction.parameters(), self.augment_classifier.parameters()),
		# 	lr=args.aux_lr, 
		# 	betas=(args.aux_beta, 0.999)
		# )

		self.train()
		self.critic_target.train()


	def train(self, training=True):
		super().train(training)
		if hasattr(self, 'augment_classifier'):
			self.augment_classifier.train(training)
			self.mi_estimator.train(training)
			self.mi_estimator_sigma.train(training)
			# self.mu_prediction.train(training)
			# self.var_prediction.train(training)

	# def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
	# 	with torch.no_grad():
	# 		_, policy_action, log_pi, _ = self.actor(next_obs)
	# 		target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
	# 		target_V = torch.min(target_Q1,
	# 							 target_Q2) - self.alpha.detach() * log_pi
	# 		target_Q = reward + (not_done * self.discount * target_V)

	# 	if self.svea_alpha == self.svea_beta:
	# 		obs = utils.cat(obs, augmentations.random_conv(obs.clone()))
	# 		action = utils.cat(action, action)
	# 		target_Q = utils.cat(target_Q, target_Q)

	# 		current_Q1, current_Q2 = self.critic(obs, action)
	# 		critic_loss = (self.svea_alpha + self.svea_beta) * \
	# 			(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
	# 	else:
	# 		current_Q1, current_Q2 = self.critic(obs, action)
	# 		critic_loss = self.svea_alpha * \
	# 			(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

	# 		obs_aug = augmentations.random_conv(obs.clone())
	# 		current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
	# 		critic_loss += self.svea_beta * \
	# 			(F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

	# 	if L is not None:
	# 		L.log('train_critic/loss', critic_loss, step)
			
	# 	self.critic_optimizer.zero_grad()
	# 	critic_loss.backward()
	# 	self.critic_optimizer.step()
	
	def compute_d2rl_classify_loss(self, x):
		loss_func = nn.BCELoss()
		# print('classify:', x.shape)
		# print(x)
		bs = x.size(0)
		# e0 = torch.zeros((int(bs/2),1), dtype=torch.float)
		# e1 = torch.ones((int(bs/2),1), dtype=torch.float)
		
		# y0 = torch.cat((e1,e0),dim=-1)
		# y1 = torch.cat((e0,e1),dim=-1)
		# y = torch.cat((y0,y1),dim=0).cuda()
		e0 = torch.zeros((bs,1), dtype=torch.float)
		e1 = torch.ones((bs,1), dtype=torch.float)
		y = torch.cat((e1,e0),dim=-1).cuda()
		return loss_func(x, y)

	def compute_d2rl_mi_loss(self, mu, logvar, sigma):
		random_index = np.random.permutation(sigma.size(0))
		shuffle_sigma = sigma[random_index]


		positive = -(mu-sigma)**2/2./torch.exp(logvar)
		negative = -(mu-shuffle_sigma)**2/2./torch.exp(logvar)
		# print((positive.sum(dim=-1)-negative.sum(dim=-1))[0])
		# print(mu.shape,logvar.shape,sigma.shape,positive.shape,negative.shape)
		sample_CLUB_bound = torch.mean(torch.sum(positive, dim=-1)-torch.sum(negative,dim=-1))
		# sample_CLUB_bound = - (-(mu - sigma)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
		# print('sample_CLUB_bound', sample_CLUB_bound)

		return torch.clamp(sample_CLUB_bound,min=0.)

	def compute_d2rl_kl_loss(self, x_mu,x_log_std, y_mu, y_log_std):
		# return F.kl_div(x.log(), y, reduction='sum')
		kl = 0.5*(y_log_std-x_log_std) - 0.5 + (torch.exp(x_log_std) + (x_mu-y_mu)**2)/2./torch.exp(y_log_std)
		
		return torch.mean(torch.sum(kl, dim=-1))

	def data_sample(self, replay_buffer):
		x = replay_buffer.sample_soda(self.soda_batch_size)
		# print(x.shape)	
		assert x.size(-1) == 100
		aug_x = x.clone()
		obs = augmentations.random_crop(x)
		aug_x = augmentations.random_crop(aug_x)
		aug_obs = augmentations.random_overlay(aug_x)

		# latent variables
		x = self.shared_cnn(obs)
		emb,sigma = self.head_cnn(x, noise=True)
		aug_x = self.shared_cnn(aug_obs)
		aug_emb,aug_sigma = self.head_cnn(aug_x, noise=True)
		return obs, aug_obs, emb, sigma, aug_emb, aug_sigma

	def update_d2rl(self, replay_buffer, L=None, step=None):
		# train mi estimator
		num_iter = 10
		for i in range(num_iter):
			# sampler.train()
			# mi_estimator.eval()
			# x_samples, y_samples = sampler.gen_samples(batch_size)
			# sampler_loss = mi_estimator(x_samples, y_samples)
			# sampler_optimizer.zero_grad()
			# sampler_loss.backward() # retain_graph=True)
			# sampler_optimizer.step()

			self.mi_estimator.train()
			self.mi_estimator_sigma.train()
			with torch.no_grad():
				_, _, emb, sigma, aug_emb, aug_sigma = self.data_sample(replay_buffer)

			mi_loss = self.mi_estimator.learning_loss(emb, sigma) + self.mi_estimator.learning_loss(aug_emb, aug_sigma)
			self.mi_optimizer.zero_grad()
			mi_loss.backward()
			nn.utils.clip_grad_norm_(self.mi_estimator.parameters(), max_norm=20, norm_type=2)
			self.mi_optimizer.step()

			mi_loss = self.mi_estimator_sigma.learning_loss(sigma, aug_sigma)
			self.mi_optimizer_sigma.zero_grad()
			mi_loss.backward()
			nn.utils.clip_grad_norm_(self.mi_estimator_sigma.parameters(), max_norm=20, norm_type=2)
			self.mi_optimizer_sigma.step()

			# torch.cuda.empty_cache()
			# print('mi_estimator', i)

		self.mi_estimator.eval()
		self.mi_estimator_sigma.eval()
		# data prepare
		obs, aug_obs, emb, sigma, aug_emb, aug_sigma = self.data_sample(replay_buffer)

		# sigma & sigma' domain classfication
		# d2rl_classify_loss = self.compute_d2rl_classify_loss(self.augment_classifier(torch.cat((aug_sigma, sigma), dim=-1)))
		# print('classify_loss:', d2rl_classify_loss)
		
		# self.d2rl_classify_optimizer.zero_grad()
		# d2rl_classify_loss.backward()
		# self.d2rl_classify_optimizer.step()

		# emb & emb' policy consistency
		# print('obs.shape', obs.shape)
		with torch.no_grad():
			act_mu, _, _, act_log_std = self.actor(obs, detach=True)
		act_mu_aug, _, _, act_log_std_aug = self.actor(aug_obs)

		# print('pi', pi, 'aug_pi', aug_pi)
		if self.kl_loss:
			kl_loss = self.compute_d2rl_kl_loss(act_mu,act_log_std, act_mu_aug, act_log_std_aug)
		else:
			kl_loss = 0.
		# print('kl_loss', kl_loss)
		
		# emb & sigma disentangled
		# mu = self.mu_prediction(emb)
		# logvar = self.var_prediction(emb)

		# aug_mu = self.mu_prediction(aug_emb)
		# aug_logvar = self.var_prediction(aug_emb)

		# mi_loss = self.compute_d2rl_mi_loss(mu, logvar, sigma) + self.compute_d2rl_mi_loss(aug_mu, aug_logvar, aug_sigma)
		if self.mi_loss:
			mi_loss = self.mi_estimator(emb, sigma) + self.mi_estimator(aug_emb, aug_sigma)
		else:
			mi_loss = 0.
		
		if self.classify_loss:
			mi_sigma_loss = self.mi_estimator_sigma(sigma, aug_sigma)
		else:
			mi_sigma_loss = 0.
		# print('mi_loss:', mi_loss)
		# print('mi_sigma_loss:', mi_sigma_loss)
		# mi_loss = 0
		# if mi_loss>0:
		# 	print('mi_loss:', mi_loss)
		# enc_loss = kl_loss + 0.1*mi_loss
		enc_loss = kl_loss + 0.1*mi_loss + 0.1*mi_sigma_loss

		

		self.encoder_optimizer.zero_grad()
		enc_loss.backward()
		nn.utils.clip_grad_norm_(itertools.chain(self.shared_cnn.parameters(),
			 				self.head_cnn.parameters()#,
			  				# self.augment_classifier.parameters()
							), max_norm=20, norm_type=2)
		self.encoder_optimizer.step()
		# torch.cuda.empty_cache()

		if L is not None:
			# L.log('train/classify_loss', d2rl_classify_loss, step)
			L.log('train/classify_loss', mi_sigma_loss, step)
			L.log('train/kl_loss', kl_loss, step)
			L.log('train/mi_loss', mi_loss, step)

		# utils.soft_update_params(
		# 	self.predictor, self.predictor_target,
		# 	self.soda_tau
		# )

	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample()

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()

		if step % self.aux_update_freq == 0:
			self.update_d2rl(replay_buffer, L, step)

# additional network temp
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0))

init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), nn.init.calculate_gain('relu'))


def apply_init_(modules):
    """
    Initialize NN modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    """
    Flatten a tensor
    """
    def forward(self, x):
        return x.reshape(x.size(0), -1)

class NonlinearOrderClassifier(nn.Module):
    def __init__(self, emb_size=256, hidden_size=4):
        super(NonlinearOrderClassifier, self).__init__()
        self.main = nn.Sequential(
            Flatten(),
            init_relu_(nn.Linear(2*emb_size, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, 2)), 
            nn.Softmax(dim=1),
        )
        self.train()

    def forward(self, emb):
        x = self.main(emb)
        return x
