import torch
from poly import UnivariatePoly
import numpy as np

import matplotlib
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
for gui in gui_env:
    try:
        # print("testing", gui)
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue
# print ("Using:",matplotlib.get_backend())
import matplotlib.pyplot as plt

if __name__ == '__main__':


    N = 100
    degree = 8

    x = torch.rand(N)*2.0 - 1.0
    y = x**3 + x**2
    xtest = torch.linspace(-1, 1, 100)
    ytest = xtest**3 + xtest**2

    model = UnivariatePoly(degree, "legendre")
    # model = UnivariatePoly(degree, "chebyshev")
    # model = UnivariatePoly(degree, "hermite")    
    
    loss_fn = torch.nn.MSELoss(size_average=False)
    learning_rate = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    nepoch = 1000
    for epoch in range(nepoch):

        X = x
        Y = y
        
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(X)[:, 0]

        # Compute and print loss.
        loss = loss_fn(y_pred, Y)
        if epoch % 100 == 0:
            print(epoch, loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

    params = list(model.parameters())
    print("final params = ", params)

    ypred = model(xtest)[:, 0]

    # plt.figure()
    # plt.plot(ypred.detach().numpy(), ytest.detach().numpy(), 'o')
    
    plt.figure()
    plt.plot(xtest.detach().numpy(), ytest.detach().numpy(), 'o')
    plt.plot(xtest.detach().numpy(), ypred.detach().numpy(), 'ro')
    
    plt.show()
