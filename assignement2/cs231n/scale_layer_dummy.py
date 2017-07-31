def scale_forward(x,a):
    out = x*a   #x is (n,d)
    cache = x,a
    return out, cache

def scale_backward(dout,cache):
    x,a = cache
    dx = dout*a
    da = np.sum(dout*x, 0)
    return dx, da
    

def relu_forward(x):
  out = np.maximum(0,x)
  cache = x
  return out, cache


def relu_backward(dout, cache):
  dx = dout
  dx[np.where(x<=0)] = 0
  return dx
