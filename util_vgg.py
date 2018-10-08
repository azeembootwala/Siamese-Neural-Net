#My comment in util should be reflected
import numpy as np


class Xavier(object):
    def __init__(self):
        pass
    def init_filter(self,shape):
        weights = np.random.randn(*shape)/np.sqrt(np.prod(shape[:-1]) + shape[-1] * np.prod(shape[:-2])/2)
        bias = np.zeros(shape[-1], dtype=np.float32)
        return weights.astype(np.float32) , bias

class Normal(object):
    def __init__(self):
        pass
    def init_filter(self, shape, pool_sz=(2,2)):
        weights = np.random.randn(*shape)/np.sqrt(np.prod(shape[:-1]) + shape[-1] * np.prod(shape[:-2])/np.prod(pool_sz))
        bias = np.zeros(shape[-1], dtype=np.float32)
        return weights.astype(np.float32) , bias
    def initialize_weights_bias(self, M1, M2):
        weights = np.random.randn(M1,M2)/np.sqrt(M1+M2)
        bias = np.zeros(M2, dtype=np.float32)
        return weights.astype(np.float32), bias.astype(np.float32)
