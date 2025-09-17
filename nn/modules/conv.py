import numpy as np
from .utils import im2col, col2im

class Convolution:
  def __init__(self, W, b, stride=1, pad=0):
    self.W = W
    self.b = b
    self.stride = stride
    self.pad = pad

    self.x = None
    self.col = None
    self.col_W = None

    self.dW = None
    self.db = None
  
  def forward(self, x):
    FN, _, FH, FW = self.W.shape
    N, _, H, W = x.shape
    out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
    out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

    col = im2col(x, FH, FW, self.stride, self.pad)
    col_W = self.W.reshape(FN, -1).T
    out = np.dot(col, col_W) + self.b

    out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

    self.x = x
    self.col = col
    self.col_W = col_W

    return out
  
  def backward(self, dout):
    FN, C, FH, FW = self.W.shape
    dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

    dcol = np.dot(dout, self.col_W.T)
    self.dW = np.dot(self.col.T, dout)
    self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
    self.db = np.sum(dout, axis=0)

    dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

    return dx
