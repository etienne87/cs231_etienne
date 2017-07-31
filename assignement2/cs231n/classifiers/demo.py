import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

weight_scale = 2e-2
learning_rate = 9e-3
model = FullyConnectedNet([100, 100],
              weight_scale=weight_scale, reg = 0.001, dtype=np.float64)
