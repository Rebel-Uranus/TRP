import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from functools import partial


def _get_out_shape_cuda(in_shape, layers):
	x = torch.randn(*in_shape).cuda().unsqueeze(0)
	return layers(x).squeeze(0).shape


def _get_out_shape(in_shape, layers):
	x = torch.randn(*in_shape).unsqueeze(0)
	return layers(x).squeeze(0).shape


def gaussian_logprob(noise, log_std):
	"""Compute Gaussian log probability"""
	residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
	return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
	"""Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
	mu = torch.tanh(mu)
	if pi is not None:
		pi = torch.tanh(pi)
	if log_pi is not None:
		log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
	return mu, pi, log_pi


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
	"""Truncated normal distribution, see https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf"""
	def norm_cdf(x):
		return (1. + math.erf(x / math.sqrt(2.))) / 2.
	with torch.no_grad():
		l = norm_cdf((a - mean) / std)
		u = norm_cdf((b - mean) / std)
		tensor.uniform_(2 * l - 1, 2 * u - 1)
		tensor.erfinv_()
		tensor.mul_(std * math.sqrt(2.))
		tensor.add_(mean)
		tensor.clamp_(min=a, max=b)
		return tensor


def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers"""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		# delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class CenterCrop(nn.Module):
	def __init__(self, size):
		super().__init__()
		assert size in {84, 100}, f'unexpected size: {size}'
		self.size = size

	def forward(self, x):
		assert x.ndim == 4, 'input must be a 4D tensor'
		if x.size(2) == self.size and x.size(3) == self.size:
			return x
		assert x.size(3) == 100, f'unexpected size: {x.size(3)}'
		if self.size == 84:
			p = 8
		return x[:, :, p:-p, p:-p]


class NormalizeImg(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x/255.


class Flatten(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)


class RLProjection(nn.Module):
	def __init__(self, in_shape, out_dim):
		super().__init__()
		self.out_dim = out_dim
		self.projection = nn.Sequential(
			nn.Linear(in_shape[0], out_dim),
			nn.LayerNorm(out_dim),
			nn.Tanh()
		)
		self.apply(weight_init)
	
	def forward(self, x):
		# print(x.shape)
		return self.projection(x)


class AdapterRLProjection(nn.Module):
	def __init__(self, in_shape, out_dim):
		super().__init__()
		self.out_dim = out_dim
		self.projection_0 = nn.Sequential(
			nn.Linear(in_shape[0], out_dim),
			nn.LayerNorm(out_dim),
			nn.Tanh()
		)
		self.projection_1 = nn.Sequential(
			nn.Linear(in_shape[0], out_dim),
			nn.LayerNorm(out_dim),
			nn.Tanh()
		)
		self.apply(weight_init)
	
	def forward(self, x):
		# print(x.shape)
		# print()
		return 0.5*(self.projection_0(x) + self.projection_1(x))


class D2RL_RLProjection(nn.Module):
	def __init__(self, in_shape, out_dim):
		super().__init__()
		self.out_dim = out_dim
		self.projection = nn.Sequential(
			nn.Linear(in_shape[0], out_dim),
			nn.LayerNorm(out_dim),
			nn.Tanh()
		)
		self.apply(weight_init)
	
	def forward(self, x):
		# x = torch.split(x, self.out_dim, dim=-1)[0]
		return self.projection(x)
	

class SODAMLP(nn.Module):
	def __init__(self, projection_dim, hidden_dim, out_dim):
		super().__init__()
		self.out_dim = out_dim
		self.mlp = nn.Sequential(
			nn.Linear(projection_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, out_dim)
		)
		self.apply(weight_init)

	def forward(self, x):
		return self.mlp(x)

class SRPMLP(nn.Module):
	def __init__(self, projection_dim, hidden_dim, out_dim):
		super().__init__()
		self.out_dim = out_dim
		self.mlp = nn.Sequential(
			nn.Linear(projection_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, out_dim)
		)
		self.apply(weight_init)

	def forward(self, x):
		return self.mlp(x)

# class SODAMLP(nn.Module):
# 	def __init__(self, projection_dim, hidden_dim, out_dim):
# 		super().__init__()
# 		self.out_dim = out_dim
# 		self.mlp = nn.Sequential(
# 			nn.Linear(projection_dim, hidden_dim),
# 			nn.BatchNorm1d(hidden_dim),
# 			nn.ReLU(),
# 			nn.Linear(hidden_dim, out_dim)
# 		)
# 		self.apply(weight_init)

# 	def forward(self, x):
# 		return self.mlp(x)


class AdapterSharedCNN(nn.Module):
	def __init__(self, obs_shape, num_layers=11, num_filters=32):
		super().__init__()
		assert len(obs_shape) == 3
		self.num_layers = num_layers
		self.num_filters = num_filters

		self.layers0_0 = nn.Sequential(CenterCrop(size=84), NormalizeImg(), nn.Conv2d(obs_shape[0], num_filters, 3, stride=2))
		self.layers1_0 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers2_0 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers3_0 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers4_0 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers5_0 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers6_0 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers7_0 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers8_0 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers9_0 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers10_0 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		# self.layers11_0 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))

		self.layers0_1 = nn.Sequential(CenterCrop(size=84), NormalizeImg(), nn.Conv2d(obs_shape[0], num_filters, 3, stride=2))
		self.layers1_1 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers2_1 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers3_1 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers4_1 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers5_1 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers6_1 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers7_1 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers8_1 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers9_1 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers10_1 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))
		# self.layers11_1 = nn.Sequential(nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1))

		x = torch.rand(32,21,21)
		self.out_shape = x.shape
		self.apply(weight_init)

	def forward(self, x):
		x0 = self.layers0_0(x)
		x1 = self.layers0_1(x)
		x = 0.5*(x0 + x1)
		x0 = self.layers1_0(x)
		x1 = self.layers1_1(x)
		x = 0.5*(x0 + x1)
		x0 = self.layers2_0(x)
		x1 = self.layers2_1(x)
		x = 0.5*(x0 + x1)
		x0 = self.layers3_0(x)
		x1 = self.layers3_1(x)
		x = 0.5*(x0 + x1)
		x0 = self.layers4_0(x)
		x1 = self.layers4_1(x)
		x = 0.5*(x0 + x1)
		x0 = self.layers5_0(x)
		x1 = self.layers5_1(x)
		x = 0.5*(x0 + x1)
		x0 = self.layers6_0(x)
		x1 = self.layers6_1(x)
		x = 0.5*(x0 + x1)
		x0 = self.layers7_0(x)
		x1 = self.layers7_1(x)
		x = 0.5*(x0 + x1)
		x0 = self.layers8_0(x)
		x1 = self.layers8_1(x)
		x = 0.5*(x0 + x1)
		x0 = self.layers9_0(x)
		x1 = self.layers9_1(x)
		x = 0.5*(x0 + x1)
		x0 = self.layers10_0(x)
		x1 = self.layers10_1(x)
		x = 0.5*(x0 + x1)
		# x0 = self.layers11_0(x)
		# x1 = self.layers11_1(x)
		# x = 0.5*(x0 + x1)
		return x


class SharedCNN(nn.Module):
	def __init__(self, obs_shape, num_layers=11, num_filters=32):
		super().__init__()
		assert len(obs_shape) == 3
		self.num_layers = num_layers
		self.num_filters = num_filters

		self.layers = [CenterCrop(size=84), NormalizeImg(), nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
		for _ in range(1, num_layers):
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(obs_shape, self.layers)
		self.apply(weight_init)

	def forward(self, x):
		return self.layers(x)


class HeadCNN(nn.Module):
	def __init__(self, in_shape, num_layers=0, num_filters=32):
		super().__init__()
		self.layers = []
		for _ in range(0, num_layers):
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers.append(Flatten())
		self.layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(in_shape, self.layers)
		# print(self.out_shape)
		self.apply(weight_init)

	def forward(self, x):
		return self.layers(x)


class VIB(nn.Module):
	def __init__(self, in_ch=2048, z_dim=256):
		super(VIB, self).__init__()
		self.in_ch = in_ch
		self.out_ch = z_dim * 2
		# self.num_class = num_class
		self.bottleneck = ChannelCompress(in_ch=self.in_ch, out_ch=self.out_ch)
		# classifier of VIB, maybe modified later.
		# classifier = []
		# classifier += [nn.Linear(self.out_ch, self.out_ch // 2)]
		# classifier += [nn.BatchNorm1d(self.out_ch // 2)]
		# classifier += [nn.LeakyReLU(0.1)]
		# classifier += [nn.Dropout(0.5)]
		# classifier += [nn.Linear(self.out_ch // 2, self.num_class)]
		# classifier = nn.Sequential(*classifier)
		# self.classifier = classifier
		# self.classifier.apply(weights_init_classifier)

	def forward(self, v):
		z_given_v = self.bottleneck(v)
		# p_y_given_z = self.classifier(z_given_v)
		# return p_y_given_z, z_given_v
		return z_given_v

class D2RL_HeadCNN(nn.Module):
	def __init__(self, in_shape, num_layers=0, num_filters=32, embed_dim=1024):
		super().__init__()
		self.layers = []
		for _ in range(0, num_layers):
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers.append(Flatten())
		# self.layers.append(nn.Linear(new_dim, int(new_dim*2)))

		# print('?')

		self.temp_layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(in_shape, self.temp_layers)
		new_dim = self.out_shape[0]

		# self.temp_layers = nn.Sequential(*self.layers)
		# self.out_shape = torch.rand(14112).shape
		# new_dim = 14112

		self.layers.append(nn.Linear(new_dim, embed_dim))
		self.layers.append(nn.ReLU())
		self.layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(in_shape, self.layers)

		self.embed = nn.Sequential(nn.Linear(embed_dim, embed_dim),
									   nn.ReLU(),
									   nn.Linear(embed_dim, embed_dim))

		self.noise = nn.Sequential(nn.Linear(embed_dim, embed_dim),
									   nn.ReLU(),
									   nn.Linear(embed_dim, embed_dim))

		# self.layers.append(nn.Linear(new_dim, int(new_dim*2)))
		# self.layers.append(nn.ReLU())
		# self.layers.append(nn.Linear(int(new_dim*2),int(new_dim*2)))
		# self.layers = nn.Sequential(*self.layers)

		# self.apply(weight_init)
		# print('??????')

	def forward(self, x, noise=False):
		x = self.layers(x)
		# print(x.shape)
		# mu,sigma = torch.split(x,int(x.size(-1)/2),dim=-1)
		embed = self.embed(x)
		if noise:
			sigma = self.noise(x)
			return embed, sigma
		return embed


class Encoder(nn.Module):
	def __init__(self, shared_cnn, head_cnn, projection):
		super().__init__()
		self.shared_cnn = shared_cnn
		self.head_cnn = head_cnn
		self.projection = projection
		self.out_dim = projection.out_dim

	def forward(self, x, detach=False):
		x = self.shared_cnn(x)
		# print(x.shape)
		x = self.head_cnn(x)
		if detach:
			x = x.detach()
		self.pro = self.projection(x)
		return self.pro

	def get_representation(self):
		return self.pro

class SDEncoder(nn.Module):
	def __init__(self, shared_cnn, head_cnn, projection):
		super().__init__()
		self.shared_cnn = shared_cnn
		self.head_cnn = head_cnn
		self.projection = projection
		self.out_dim = projection.out_dim

	def forward(self, x, detach=False):
		x = self.shared_cnn(x)
		x = self.head_cnn(x)
		if detach:
			x = x.detach()
		
		self.pro = self.projection(x)
		return self.pro

	def get_representation(self):
		return self.pro

class SDACEncoder(nn.Module):
	def __init__(self, shared_cnn, head_cnn, bottleneck, projection):
		super().__init__()
		self.shared_cnn = shared_cnn
		self.head_cnn = head_cnn
		self.bottleneck = bottleneck
		self.projection = projection
		self.out_dim = projection.out_dim

	def forward(self, x, detach=False):
		x = self.shared_cnn(x)
		x = self.head_cnn(x)
		x = self.bottleneck(x)
		if detach:
			x = x.detach()
		
		self.pro = self.projection(x)
		return self.pro
	
	def get_representation(self):
		return self.pro

class SRPEncoder(nn.Module):
	def __init__(self, encoedr, action_dim, projection_dim, hidden_dim, out_dim, concate_dim):
		super().__init__()
		self.out_dim = out_dim
		self.obs_encoedr = encoedr
		self.act_encoder = nn.Sequential(
			nn.Linear(action_dim, hidden_dim),
			nn.LayerNorm(out_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, projection_dim),
			nn.Tanh(),
		)
		self.mlp = nn.Sequential(
			nn.Linear(concate_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)
		

	def forward(self, obs, actions):
		x = self.shared_cnn(x)
		x = self.head_cnn(x)
		if detach:
			x = x.detach()
		
		self.pro = self.projection(x)
		return self.pro
	


class Actor(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim, log_std_min, log_std_max):
		super().__init__()
		self.encoder = encoder
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		self.mlp = nn.Sequential(
			nn.Linear(self.encoder.out_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 2 * action_shape[0]),
		)
		self.mlp.apply(weight_init)

	def forward(
		self,
		x,
		compute_split_log_pi=False,
		mask=False,
		compute_pi=True,
		compute_log_pi=True,
		detach=False,
		compute_attrib=False,
	):
		# if mask is not None:
		#	 x = self.encoder(x, mask, detach)
		# else:
		if not mask:
			x = self.encoder(x, detach)
		mu, log_std = self.mlp(x).chunk(2, dim=-1)
		log_std = torch.tanh(log_std)
		log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
			log_std + 1
		)

		if compute_pi:
			std = log_std.exp()
			noise = torch.randn_like(mu)
			pi = mu + noise * std
		else:
			pi = None
			entropy = None

		if compute_log_pi:
			log_pi = gaussian_logprob(noise, log_std)
			# print(mu.shape, pi.shape, log_pi.shape)
			if compute_split_log_pi:
				split_log_pi = (-0.5 * noise.pow(2) - log_std)
				split_log_pi =  split_log_pi - 0.5 * np.log(2 * np.pi) * noise.size(-1)
		else:
			log_pi = None

		
		mu, pi, log_pi = squash(mu, pi, log_pi)

		if compute_split_log_pi:
			split_log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
			return mu, pi, log_pi, log_std, split_log_pi
		return mu, pi, log_pi, log_std

	def get_y_given_rep(self):
		return self.mu, self.log_std

class QFunction(nn.Module):
	def __init__(self, obs_dim, action_dim, hidden_dim):
		super().__init__()
		self.trunk = nn.Sequential(
			nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)
		self.apply(weight_init)

	def forward(self, obs, action):
		assert obs.size(0) == action.size(0)
		return self.trunk(torch.cat([obs, action], dim=1))


class Critic(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.Q1 = QFunction(
			self.encoder.out_dim, action_shape[0], hidden_dim
		)
		self.Q2 = QFunction(
			self.encoder.out_dim, action_shape[0], hidden_dim
		)

	def forward(self, x, action, detach=False):
		x = self.encoder(x, detach)
		self.q1_predict = self.Q1(x, action)
		self.q2_predict = self.Q2(x, action)
		return self.q1_predict, self.q2_predict
	
	def get_y_given_rep(self):
		return self.q1_predict, self.q2_predict


class CURLHead(nn.Module):
	def __init__(self, encoder):
		super().__init__()
		self.encoder = encoder
		self.W = nn.Parameter(torch.rand(encoder.out_dim, encoder.out_dim))

	def compute_logits(self, z_a, z_pos):
		"""
		Uses logits trick for CURL:
		- compute (B,B) matrix z_a (W z_pos.T)
		- positives are all diagonal elements
		- negatives are all other elements
		- to compute loss use multiclass cross entropy with identity matrix for labels
		"""
		Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
		logits = torch.matmul(z_a, Wz)  # (B,B)
		b = logits.size()[0]
		logits = torch.cat((logits[:,:b],torch.diag(logits[:,b:]).reshape(-1,1)),dim=1)
		logits = logits - torch.max(logits, 1)[0][:, None]
		return logits


class InverseDynamics(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.mlp = nn.Sequential(
			nn.Linear(2*encoder.out_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, action_shape[0])
		)
		self.apply(weight_init)

	def forward(self, x, x_next):
		h = self.encoder(x)
		h_next = self.encoder(x_next)
		joint_h = torch.cat([h, h_next], dim=1)
		return self.mlp(joint_h)


class SODAPredictor(nn.Module):
	def __init__(self, encoder, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.mlp = SODAMLP(
			encoder.out_dim, hidden_dim, encoder.out_dim
		)
		self.apply(weight_init)

	def forward(self, x):
		return self.mlp(self.encoder(x))


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
	'''
		This class provides the CLUB estimation to I(X,Y)
		Method:
			forward() :	  provides the estimation with input samples  
			loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
		Arguments:
			x_dim, y_dim :		 the dimensions of samples from X, Y respectively
			hidden_size :		  the dimension of the hidden layer of the approximation network q(Y|X)
			x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
	'''
	def __init__(self, x_dim, y_dim, hidden_size):
		super(CLUB, self).__init__()
		# p_mu outputs mean of q(Y|X)
		#print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
		self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
									   nn.ReLU(),
									   nn.Linear(hidden_size//2, y_dim))
		# p_logvar outputs log of variance of q(Y|X)
		self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
									   nn.ReLU(),
									   nn.Linear(hidden_size//2, y_dim),
									   nn.Tanh())

	def get_mu_logvar(self, x_samples):
		mu = self.p_mu(x_samples)
		logvar = self.p_logvar(x_samples)
		return mu, logvar
	
	def forward(self, x_samples, y_samples): 
		mu, logvar = self.get_mu_logvar(x_samples)
		
		# log of conditional probability of positive sample pairs
		positive = - (mu - y_samples)**2 /2./logvar.exp()  
		
		prediction_1 = mu.unsqueeze(1)		  # shape [nsample,1,dim]
		y_samples_1 = y_samples.unsqueeze(0)	# shape [1,nsample,dim]

		# log of conditional probability of negative sample pairs
		negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

		return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

	def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
		mu, logvar = self.get_mu_logvar(x_samples)
		return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
	
	def learning_loss(self, x_samples, y_samples):
		return - self.loglikeli(x_samples, y_samples)

class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
	def __init__(self, x_dim, y_dim, hidden_size):
		super(CLUBSample, self).__init__()
		self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
									   nn.ReLU(),
									   nn.Linear(hidden_size//2, y_dim))

		self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
									   nn.ReLU(),
									   nn.Linear(hidden_size//2, y_dim),
									   nn.Tanh())

	def get_mu_logvar(self, x_samples):
		mu = self.p_mu(x_samples)
		logvar = self.p_logvar(x_samples)
		return mu, logvar
	 
		
	def loglikeli(self, x_samples, y_samples):
		mu, logvar = self.get_mu_logvar(x_samples)
		return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
	

	def forward(self, x_samples, y_samples):
		mu, logvar = self.get_mu_logvar(x_samples)
		
		sample_size = x_samples.shape[0]
		#random_index = torch.randint(sample_size, (sample_size,)).long()
		random_index = torch.randperm(sample_size).long()
		
		positive = - (mu - y_samples)**2 / logvar.exp()
		negative = - (mu - y_samples[random_index])**2 / logvar.exp()
		upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
		res = torch.clamp(upper_bound,min=0.)

		return res/2.

	def learning_loss(self, x_samples, y_samples):
		return - self.loglikeli(x_samples, y_samples)


class ChannelCompress(nn.Module):
	def __init__(self, in_ch=2048, out_ch=256):
		"""
		reduce the amount of channels to prevent final embeddings overwhelming shallow feature maps
		out_ch could be 512, 256, 128
		"""
		super(ChannelCompress, self).__init__()
		num_bottleneck = 1000
		add_block = []
		add_block += [nn.Linear(in_ch, num_bottleneck)]
		# add_block += [nn.BatchNorm1d(num_bottleneck)]
		add_block += [nn.ReLU()]

		add_block += [nn.Linear(num_bottleneck, 500)]
		# add_block += [nn.BatchNorm1d(500)]
		add_block += [nn.ReLU()]
		add_block += [nn.Linear(500, out_ch)]

		# Extra BN layer, need to be removed
		#add_block += [nn.BatchNorm1d(out_ch)]

		add_block = nn.Sequential(*add_block)
		add_block.apply(weights_init_kaiming)
		self.model = add_block

	def forward(self, x):
		x = self.model(x)
		return x

def weights_init_kaiming(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
		init.constant_(m.bias.data, 0.0)
	elif classname.find('BatchNorm1d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)