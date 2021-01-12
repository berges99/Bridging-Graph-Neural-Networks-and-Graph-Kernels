import copy
import torch

from tqdm import tqdm
from torch import optim, nn



class ZeroOutput(nn.Module):
	'''Zero out the output of a model by subtracting out a copy of it.'''
	def __init__(self, model):
		super().__init__()
		self.model = model
		self.init_model = copy.deepcopy(model).to('cuda:0')

	def forward(self, x):
		return self.model(x) - self.init_model.eval()(x)


class Scale(nn.Module):
	'''Scales the output of a model by parameter alpha.'''
	def __init__(self, model, alpha):
		super().__init__()
		self.model = model
		self.alpha = alpha

	def forward(self, x):
		return self.alpha * self.model(x)


def MLP(input_size,
		output_size,
		hidden_size,
		hidden_layers=1,
		bias=True,
		zero_output=True,
		alpha=None):
	'''Simple Multi-Layer-Perceptron (MLP) with ReLU activation functions.

	Parameters:
		- input_size: (int) size of the input
		- output_size: (int) size of the output
		- hidden_size: (int) width of the hidden layers 
		- bias: (bool) whether to include biases
		- zero_output: (bool) whether to zero out the output of the model
		- alpha: (float) scale of the output
		- hidden_layers: (int) number of hidden layers

	Returns:
		- (nn.Module) model

	'''
	model = nn.Sequential(
		nn.Linear(input_size, hidden_size, bias=bias),
		nn.ReLU(),
		*[
			layer 
			for _ in range(hidden_layers - 1) 
			for layer in [nn.Linear(hidden_size, hidden_size, bias=bias), nn.ReLU()]
		],
		nn.Linear(hidden_size, 1, bias=bias)
	)
	if zero_output:
		model = ZeroOutput(model)
	if alpha is not None:
		model = Scale(model, alpha)
	return model


def GradientDescent(model, x, y, iters=100, lr=1e-03, eps=1e-10, alpha=1.0):
	'''Run stochastic gradient descent using square loss on the model with the given data.
	   It also updates the model instance.

	Parameters:
		- model: (nn.Module)
		- x: (torch.tensor)
		- y: (torch.tensor)
		- iters: (int)
		- lr: (float)
		- eps: (float) lower bound for training loss
		- alpha: (float) scaling/normalizing factor. The loss value is divided by alpha**2
		
	Returns:
		- (list) loss values (unscaled)
	
	'''
	# Init the pytorch SGD optimizer
	opt = optim.SGD(model.parameters(), lr=lr)
	# Return loss values
	losses = []
	# Start training with gradient descent
	loss_item = -1
	for i in range(iters):
		# Compute model outputs
		out = model(x)
		# Normalize the loss
		loss = 1 / (alpha**2) * nn.MSELoss()(out, y)
		# Store the unnormalized losses
		loss_item = loss.item() * alpha**2
		losses.append(loss_item)
		# Check stopping conditions
		if loss_item < eps:	return losses
		opt.zero_grad()
		loss.backward()
		opt.step()
	return losses


def NTK(model, x, device):
	'''Compute the Neural Tangent Kernel of the model on the inputs x.

	Parameters:
		- model: (nn.Module) model to feed the input to
		- x: (torch.Tensor) input to the model

	Returns:
		- (torch.Tensor) gradient feature map
		- (torch.Tensor) tangent kernel

	'''
	# Forward the inputs to the model
	out = model(x)
	p, = nn.utils.parameters_to_vector(model.parameters()).shape
	n, out_dim = out.shape
	# Transposed Jacobian of the model
	features = torch.zeros(n, p, requires_grad=False).to(device)
	# Loop over data points
	for i in range(n):
		model.zero_grad()
		out[i].backward(retain_graph=True)
		p_grad = torch.tensor([], requires_grad=False).to(device)
		for p in model.parameters():
			p_grad = torch.cat((p_grad, p.grad.reshape(-1)))
		features[i, :] = p_grad
	# Matrix multiplication to obtain the tangent kernel
	ntk = features @ features.t()
	return features, ntk
