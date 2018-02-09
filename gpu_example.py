import torch
from torch.autograd import Variable
from poly import Legendre, MultiLegendre
import numpy as np
import time


def gpu_test():
    dtype = torch.cuda.FloatTensor
    
    x = Variable(torch.linspace(-1, 1, 100).type(dtype))
    order = 30

    ptorch = Legendre(order)
    vand_torch = ptorch(x)
    print("torch done")

    x = x.cpu()
    vand = np.polynomial.legendre.legvander(x.data.numpy(), order)

    vand_torch = vand_torch.cpu()
    
    difference = np.linalg.norm(vand - vand_torch.data.numpy())
    print("difference = ", difference)

    assert difference < 1e-5, "pytorch and numpy.legendre not same"


def gpu_time(N, order):

    dtype = torch.cuda.FloatTensor
    x = Variable(torch.linspace(-1, 1, N).type(dtype))

    ptorch = Legendre(order)
    vand_torch = ptorch(x)

    return vand_torch

def cpu_time(N, order):

    dtype = torch.FloatTensor
    x = Variable(torch.linspace(-1, 1, N).type(dtype))

    ptorch = Legendre(order)
    vand_torch = ptorch(x)

    return vand_torch

def numpy_time(N , order):

    xnump = np.linspace(-1, 1, N)
    vand = np.polynomial.legendre.legvander(xnump, order)
    return vand
    

if __name__ == "__main__":
    

    # gpu_test()

    N = 1000000
    order = 200
    
    start = time.clock()
    gpu_vand = gpu_time(N, order)
    end = time.clock()
    print("GPU Elapsed time = ", end - start)

    start = time.clock()
    cpu_vand = cpu_time(N, order)
    end = time.clock()
    print("CPU Elapsed time = ", end - start)

    start = time.clock()
    numpy_vand = numpy_time(N, order)
    end = time.clock()
    print("Numpy Elapsed time = ", end - start)

    
    diff = torch.norm(gpu_vand.cpu() - cpu_vand) / torch.norm(cpu_vand)
    print("diff = ", diff)
