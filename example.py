import torch
from torch.autograd import Variable
import poly
import numpy as np
import time


def uni_test():

    dtype = torch.DoubleTensor
    
    x = torch.linspace(-1, 1, 10000).type(dtype)
    order = 300
    vand = np.polynomial.legendre.legvander(x.data.numpy(), order)

    # print("vand done", vand.shape)

    # def func():
    # ptorch = Legendre(order)
    vand_torch = poly.legendre(x, order)
    print("torch done")

    difference = np.linalg.norm(vand - vand_torch.data.numpy())
    print("difference = ", difference)

    # checking to make sure can run multiple times
    vand_torch = poly.legendre(x, order)
    vand_torch = poly.legendre(x, order)
    difference = np.linalg.norm(vand - vand_torch.data.numpy())
    print("difference = ", difference)

    assert difference < 1e-5, "pytorch and numpy.legendre not same"

def uni_time():
    dtype = torch.DoubleTensor
    N = 10000
    xnump = np.linspace(-1, 1, N)
    x = Variable(torch.linspace(-1, 1, N).type(dtype))
    order = 400
    def fnump():
        vand = np.polynomial.legendre.legvander(xnump, order)
        return vand

    def ftorch():
        vand_torch = poly.legendre(x, order)
        return vand_torch


    print("fnump ")
    start = time.clock()
    for ii in range(100):
        fnump()
    end = time.clock()
    print("Elapsed time = ", end - start)


    print("ftorch ")
    start = time.clock()
    for ii in range(100):
        ftorch()
    end = time.clock()
    print("Elapsed time = ", end - start)
    
    
if __name__ == "__main__":

    uni_test()
    uni_time()

    dim = 20
    order = 3
    N = 40
    x = Variable(torch.rand(N, dim)*2.0 - 1.0)

    # poly = MultiLegendre(dim, order)
    # print("Number of unknowns = ", poly.num)
    # print(poly(x))

    # for param in poly.parameters():
    #     print(param)
    
