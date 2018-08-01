""" PyTorchPoly """

import torch
from torch.autograd import Variable
import torch.nn as nn

# import pyindex

def legendre(x, degree):
    retvar = torch.ones(x.size(0), degree+1).type(x.type())
    # retvar[:, 0] = x * 0 + 1
    if degree > 0:
        retvar[:, 1] = x
        for ii in range(1, degree):
            retvar[:, ii+1] = ((2 * ii + 1) * x * retvar[:, ii] - \
                               ii * retvar[:, ii-1]) / (ii + 1)
    return retvar

def chebyshev(x, degree):
    retvar = torch.zeros(x.size(0), degree+1).type(x.type())
    retvar[:, 0] = x * 0 + 1
    if degree > 0:
        retvar[:, 1] = x
        for ii in range(1, degree):
            retvar[:, ii+1] = 2 * x * retvar[:, ii] -  retvar[:, ii-1]

    return retvar

def hermite(x, degree):
    retvar = torch.zeros(x.size(0), degree+1).type(x.type())
    retvar[:, 0] = x * 0 + 1
    if degree > 0:
        retvar[:, 1] = x
        for ii in range(1, degree):
            retvar[:, ii+1] = x * retvar[:, ii] - retvar[:, ii-1] / ii

    return retvar

class UnivariatePoly(nn.Module):
    """ Univariate Legendre Polynomial """
    def __init__(self, PolyDegree, poly_type):
        super(UnivariatePoly, self).__init__()
        self.degree = PolyDegree
        self.linear = nn.Linear(PolyDegree+1, 1, bias=False)
        self.poly_type = poly_type

    def forward(self, x):

        if self.poly_type == "legendre":
            vand = legendre(x, self.degree)
        elif self.poly_type == "chebyshev":
            vand = chebyshev(x, self.degree)
        elif self.poly_type == "hermite":
            vand = hermite(x, self.degree)            
        else:
            print("No Polynomial type ", self.poly_type, " is implemented")
            exit(1)
        # print("vand = ", vand)
        retvar = self.linear(vand)

        return retvar

# class MultiLegendre(nn.Module):

#     def __init__(self, dim, order):
#         super(MultiLegendre, self).__init__()
#         self.order = order
#         self.dim = dim
#         self.num = pyindex.TotalOrder(self.dim, self.order).get_count()
#         self.polys = nn.ModuleList([Legendre(order) for i in range(dim)])

#     def forward(self, x):
#         vs = [p(x[:, ii]) for ii, p in enumerate(self.polys)]
#         v = Variable(torch.ones((x.size(0), self.num)).type(x.data.type()))
#         iterator = pyindex.TotalOrder(self.dim, self.order)
#         for ii, inds in enumerate(iterator):
#             for jj, ind in enumerate(inds):
#                 v[:,ii] = v[:, ii] * vs[jj][:, ind]
                
#         return v

#     def num_parameters(self):
#         return self.num
