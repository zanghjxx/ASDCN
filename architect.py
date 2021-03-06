import torch
import numpy as np
import torch.nn as nn
from option import args
from torch.autograd import Variable

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, loss, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.loss = loss
    self.optimizer = torch.optim.Adam(self.model.model.change_arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer,idx_scale):
    output = self.model(input, idx_scale)[0]
    loss = self.loss(output, target)
    theta = _concat(self.model.model.parameters())
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)).data + self.network_weight_decay*theta

    unrolled_model = self._construct_model_from_theta(theta.sub((moment+dtheta)*eta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, idx_scale, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer,idx_scale)
    else:
        self._backward_step(input_valid, target_valid, idx_scale)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid, idx_scale):
    self.model.training = False
    output = self.model(input_valid, idx_scale)[0]
    loss = self.loss(output, target_valid)
    loss.backward()
    self.model.training = True

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer,idx_scale):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer,idx_scale)
    output = unrolled_model(input_valid)[0]
    unrolled_loss = self.loss(output, target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.change_arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train, idx_scale)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.model.change_arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.model.new(args)
    model_dict = self.model.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, idx_scale, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    output = self.model(input,idx_scale)[0]
    loss = self.loss(output, target)
    grads_p = torch.autograd.grad(loss, self.model.model.change_arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    output = self.model(input,idx_scale)[0]
    loss = self.loss(output, target)
    grads_n = torch.autograd.grad(loss, self.model.model.change_arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
