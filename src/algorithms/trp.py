import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import itertools
import utils
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC
import time

### VIT ###
class TRP(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.svea_alpha = args.svea_alpha
		self.svea_beta = args.svea_beta
		self.aux_update_freq = args.aux_update_freq
		self.jump = args.jump
		self.srp_batch = 256

		self.gammga_list = torch.logspace(0, self.jump-1, steps=self.jump, base=args.discount).cuda()
		shared_cnn = self.critic.encoder.shared_cnn
		head_cnn = self.critic.encoder.head_cnn

		num_heads = 4
		
		self.obs_encoder = m.Encoder(
			shared_cnn,
			head_cnn,
			m.RLProjection(head_cnn.out_shape, args.projection_dim)
		).cuda()
		
		self.act_encoder = nn.Sequential(
			nn.Linear(action_shape[0], args.projection_dim),
		).cuda()
		
		self.transformer = nn.ModuleList([
			Block(args.projection_dim, num_heads, mlp_ratio=2., 
					qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
					drop_path=0., init_values=0., act_layer=nn.GELU, 
					norm_layer=nn.LayerNorm, attn_head_dim=None) 
			for _ in range(args.num_attn_layers)]).cuda()
		
		self.position = PositionalEmbedding(args.projection_dim)
		
		self.reg_token = RegToken(args.projection_dim)
		
		self.predictor = nn.Sequential(
			# nn.LayerNorm(args.projection_dim),
			nn.Linear(args.projection_dim, args.projection_dim),
			nn.ReLU(),
			nn.Linear(args.projection_dim, args.projection_dim),
			nn.ReLU(),
			nn.Linear(args.projection_dim, 1)
		).cuda()


		self.srp_optimizer = torch.optim.Adam(
			itertools.chain(self.obs_encoder.parameters(), self.act_encoder.parameters(), self.transformer.parameters(), self.predictor.parameters(), self.reg_token.parameters()),
			lr=args.aux_lr,
			betas=(args.aux_beta, 0.999)
		)
		
	def train(self, training=True):
		super().train(training)
		if hasattr(self, 'predictor'):
			self.obs_encoder.train(training)
			self.act_encoder.train(training)
			self.transformer.train(training)
			self.predictor.train(training)



	def compute_segment_return_predict_loss(self, obs, action):
		B = obs.size(0)
		T = self.jump + 1 + 1
		obs_embed = self.obs_encoder(obs).unsqueeze(1)
		act_embed = self.act_encoder(action.flatten(0,1)).view(B, self.jump, -1)
		position = self.position(T).transpose(0, 1).cuda() # (1, T, Z) -> (T, 1, Z)
		expand_pos_emb = position.expand(T, B, -1).permute(1, 0, 2)  # (T, B, Z)
		
		reg_token = self.reg_token().expand(B, 1, -1)

		x = torch.cat((reg_token, obs_embed, act_embed), dim=1)
		x = (x + expand_pos_emb)
		for i in range(len(self.transformer)):
			x = self.transformer[i](x)
		
		return self.predictor(x[:,0,:])
	
	def update_aux(self, replay_buffer, L=None, step=None):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_btsac(self.srp_batch)
		
		obs = obs[:,0,:,:,:]
		reward = utils.cat(reward, reward)
		score = reward.squeeze() * self.gammga_list

		obs = obs.reshape(-1, 9, 84, 84)

		score = score.reshape(self.srp_batch*2, -1)
		score = score.sum(dim=-1, keepdim=True)

		obs = utils.cat(obs, augmentations.random_overlay(obs.clone()))
		action = utils.cat(action, action)
		predict_score = self.compute_segment_return_predict_loss(obs, action)
		aux_loss = F.mse_loss(predict_score, score)
		
		self.srp_optimizer.zero_grad()
		aux_loss.backward()
		self.srp_optimizer.step()
		if L is not None:
			L.log('train/aux_loss', aux_loss, step)

	

	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_svea(n=256)

		# obs = utils.cat(obs, augmentations.random_overlay(obs.clone()))
		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()
		
		if step % self.aux_update_freq == 0:
			self.update_aux(replay_buffer, L, step)


##################
class RegToken(nn.Module):

	def __init__(self, d):
		super().__init__()

		self.reg_token = nn.Parameter(torch.randn(1, 1, d)).cuda()

	def forward(self):
		return self.reg_token


class PositionalEmbedding(nn.Module):

	def __init__(self, d_model, max_len=128):
		super().__init__()

		pe = torch.zeros(max_len, d_model).float()
		pe.require_grad = False

		position = torch.arange(0, max_len).float().unsqueeze(1)
		div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, length):
		return self.pe[:, :length]


class Block(nn.Module):

	def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
				 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
				 attn_head_dim=None):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.attn = Attention(
			dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
			attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
		
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

		if init_values > 0:
			self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
			self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
		else:
			self.gamma_1, self.gamma_2 = None, None

	def forward(self, x):
		if self.gamma_1 is None:
			x = x + self.drop_path(self.attn(self.norm1(x)))
			x = x + self.drop_path(self.mlp(self.norm2(x)))
		else:
			x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
			x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
		return x


class DropPath(nn.Module):
	def __init__(self, drop_prob=None):
		super(DropPath, self).__init__()
		self.drop_prob = drop_prob

	def forward(self, x):
		return drop_path(x, self.drop_prob, self.training)
	
	def extra_repr(self) -> str:
		return 'p={}'.format(self.drop_prob)


class Attention(nn.Module):
	def __init__(
			self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
			proj_drop=0., attn_head_dim=None):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		if attn_head_dim is not None:
			head_dim = attn_head_dim
		all_head_dim = head_dim * self.num_heads
		self.scale = qk_scale or head_dim ** -0.5

		self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
		if qkv_bias:
			self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
			self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
		else:
			self.q_bias = None
			self.v_bias = None

		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(all_head_dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x):
		
		B, N, C = x.shape
		qkv_bias = None
		if self.q_bias is not None:
			qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
		qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
		qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2] 
		
		q = q * self.scale
		attn = (q @ k.transpose(-2, -1))
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
		x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x

class Mlp(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(drop)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.fc2(x)
		x = self.drop(x)
		return x