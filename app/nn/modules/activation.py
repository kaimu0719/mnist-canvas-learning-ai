import numpy as np
from utils import softmax, cross_entropy_error

class ReLU:
  def __init__(self):
    self.mask = None

  def forward(self, x):
    self.mask = (x <= 0)
    out = x.copy()
    out[self.mask] = 0

    return out
  
  def backward(self, dout):
    dout[self.mask] = 0
    dx = dout

    return dx
  
class SoftmaxWithLoss:
  def __init__(self):
    self.loss = None
    self.y = None
    self.t = None
  
  def forward(self, y, t):
    self.t = t
    self.y = softmax(y)
    self.loss = cross_entropy_error(self.y, self.t)

    return self.loss
  
  def backward(self, dout=1):
    batch_size = self.y.shape[0]
    if self.y.size == self.t.size:
      dx = (self.y - self.t) / batch_size
    else:
      dx = self.y.copy()
      dx[np.arange(batch_size), self.t] -= 1
      dx /= batch_size
    
    return dx * dout