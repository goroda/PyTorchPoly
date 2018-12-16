import torch
from torch.autograd import Variable
import poly
import numpy as np
import time

torch.set_num_threads(4)

def gpu_test():
    dtype = torch.cuda.FloatTensor
    
    x = torch.linspace(-1, 1, 100).type(dtype)
    order = 30

    vand_torch = poly.legendre(x, order)
    print("torch done")

    x = x.cpu()
    vand = np.polynomial.legendre.legvander(x.data.numpy(), order)
    vand_torch = vand_torch.cpu()
    
    difference = np.linalg.norm(vand - vand_torch.data.numpy())
    print("difference = ", difference)

    assert difference < 1e-5, "pytorch and numpy.legendre not same"


def gpu_time(N, order):

    dtype = torch.cuda.FloatTensor
    x = torch.linspace(-1, 1, N).type(dtype)

    vand_torch = poly.legendre(x, order)
    return vand_torch

# def gpu_time_nd(order, dim, x):

#     dtype = torch.cuda.FloatTensor
#     ptorch = MultiLegendre(dim, order)
#     vand_torch = ptorch(Variable(x.type(dtype)))

#     return vand_torch


def cpu_time(N, order):

    dtype = torch.FloatTensor
    x = torch.linspace(-1, 1, N).type(dtype)
    vand_torch = poly.legendre(x, order)

    return vand_torch

# def cpu_time_nd(order, dim, x):

#     dtype = torch.FloatTensor
#     ptorch = MultiLegendre(dim, order)
#     vand_torch = ptorch(Variable(x.type(dtype)))

#     return vand_torch

def numpy_time(N , order):

    xnump = np.linspace(-1, 1, N)
    vand = np.polynomial.legendre.legvander(xnump, order)
    return vand
    

if __name__ == "__main__":
    

    gpu_test()

    N = 1000000
    order = 200

    print("Univariate ")
    
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
    

    # print("Multivariate (warning on CPU takes almost a minute )")
    # dim = 10
    # order = 5
    # N = 100000
    # x = torch.rand(N, dim) * 2.0 - 1.0

    
    # start = time.clock()
    # gpu_vand = gpu_time_nd(order, dim, x)
    # end = time.clock()
    # print("Multivariate GPU Elapsed time = ", end - start)


    # start = time.clock()
    # cpu_vand = cpu_time_nd(order, dim, x)
    # end = time.clock()
    # print("Multivariate CPU Elapsed time = ", end - start)

    # diff = torch.norm(gpu_vand.cpu() - cpu_vand) / torch.norm(cpu_vand)
    # print("diff = ", diff)
