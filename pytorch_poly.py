import torch
from torch.autograd import Variable
import torch.nn as nn

# sys.path.append("/Users/alex/Software/mypython/pyindex")
import pyindex

class Legendre(nn.Module):

    def __init__(self, order):
        super(Legendre, self).__init__()
        self.order = order
    
    def forward(self, x):

        retvar = Variable(torch.zeros(x.size(0), self.order+1).type(x.data.type()))
        p2 = Variable(torch.ones(x.size()).type(x.data.type()))
        p1 = x
        retvar[:, 0] = p2
        if self.order == 1:
            retvar[:, 1] = p1
        elif self.order > 1:
            retvar[:, 0] = p2
            retvar[:, 1] = p1
            for ii in range(1, self.order):
                retvar[:, ii+1] = ((2 * ii + 1) * x * retvar[:, ii] - \
                                   ii * retvar[:, ii-1]) / (ii + 1)
                            
        return retvar
    
class MultiLegendre(nn.Module):

    def __init__(self, dim, order):
        super(MultiLegendre, self).__init__()
        self.order = order
        self.dim = dim
        self.num = pyindex.TotalOrder(self.dim, self.order).get_count()
        self.polys = nn.ModuleList([Legendre(order) for i in range(dim)])

    def forward(self, x):
        vs = [p(x[:, ii]) for ii, p in enumerate(self.polys)]
        v = Variable(torch.ones((x.size(0), self.num)).type(x.data.type()))
        iterator = pyindex.TotalOrder(self.dim, self.order)
        for ii, inds in enumerate(iterator):
            for jj, ind in enumerate(inds):
                v[:,ii] = v[:, ii] * vs[jj][:, ind]
                
        return v

    def num_parameters(self):
        return self.num
