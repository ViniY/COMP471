import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rng


def groundTruth(input):
    #     out = np.sin(16*input)  *  np.exp(-5*input**2)
    #     out = input**3
    #     out = input -0.2
    out = np.tanh(10 * input)
    return (out)


def rngWeights(M, sigmaWeights, sigmaBiases):  # make up the random parameters for the input-to-hidden linear mapping.
    W = rng.normal(0, sigmaWeights, (M))  # yes, weird in that there's only 1 input dimension, built in. Sucks...
    B = rng.normal(0, sigmaBiases, (M))
    return (W, B)


def doDesignMatrix(W, B, x, active_func):  # projection of x, using those W,B and a
    # nonlinearity: Bishop denotes this Phi

    if active_func == 1:
        Phi = np.maximum(0, np.outer(x, W) + B)  # the nonlinearity here is ReLU.

    else:
        hs = []
        R = sigmaWeights / 2
        for i in x:
            distance = 0
            for j in W:
                if active_func == 2:
                    hs.append(np.exp(-((np.sqrt((i - j) ** 2)) ** 2) / R ** 2))  # exp^(-r^2/R^2)
                if active_func == 3:
                    hs.append((np.sqrt((i - j) ** 2)) ** 3)
        if active_func == 2 or active_func == 3:
            h = np.asarray(hs)
            h = h.reshape(len(x), len(W))
            Phi = h
    #     print("distance shape : ", np.shape(h))
    #
    # print("The shape of Phi :", np.shape(Phi))

    return (Phi)

M = 100  # number of basis functions (a.k.a. hidden units)
N = 1 # number of data items
Ntest = 100  # number of points to plot when showing the plotted lines

xData = 0.5*(1-2*rng.random((N,1)))
# xData = 0.5 * np.array([-1, -.9, -.8, -.7, -.6, 1, .9, .8, .7, .6]).reshape(N, 1)

yData = groundTruth(xData)
x = np.linspace(-1, 1, Ntest)  # .reshape(Ntest,1)

# For the hid-to-output mapping:
alpha = 1.0
beta = 10000.0

# For the input-to-hidden mapping:
sigmaWeights = 1.0
sigmaBiases = 1.0


# %% md

### Here is the magic linalg incantation

# %%

def BayesLinearRegression(W, B, alpha, beta, xData, xQuery,active_func):
    Phi = doDesignMatrix(W, B, xData,active_func)
    invS = alpha * np.eye(M) + beta * np.dot(Phi.transpose(), Phi)
    S = np.linalg.inv(invS)
    regMoorePenrose = beta * np.dot(S, Phi.transpose())  # nb. the beta here "undoes" the one inside invS
    wLMS = np.dot(regMoorePenrose, yData)
    phi = doDesignMatrix(W, B, xQuery,active_func)
    y = np.dot(phi, wLMS)
    sig2 = 1 / beta + np.diag(np.dot(np.dot(phi, S), phi.transpose())).reshape(y.shape)
    return (y, sig2)


fig=plt.figure(figsize=(18, 6))
active_func = 1 #RBF increasing with distance
# LHS, RHS = plt.subplot(121), plt.subplot(122)

W,B = rngWeights(M,sigmaWeights,sigmaBiases)
y_s=[]
plot = False
for i in range(50):
    if i < 10:
        if i % 2== 0:
            plot = True
        else:
            plot = False
    elif i % 5 == 0:
        plot = True
    else:
        plot = False


    xData = np.append(xData, (0.5 * (1 - 2 * rng.random((N, 1)))))
    yData = groundTruth(xData)
    y, sig2 = BayesLinearRegression(W,B,alpha,beta,xData,x,active_func)
    upper = np.ravel(y + 2*np.sqrt(sig2))
    lower = np.ravel(y - 2*np.sqrt(sig2))

    if plot:
        LHS = plt.subplot()
        LHS.fill_between(np.ravel(x), lower, upper, facecolor='pink', interpolate=True, alpha=.5)
        LHS.plot(x, y, '-r', alpha=1)
        LHS.set_ylim(-2, 2)
        LHS.plot(x, groundTruth(x), ':g', xData, yData, 'ok')
        print(i)
        plt.show()

# # for the RHS, we just draw samples i.i.d at each x value in linspace
# for i in range(200):
#     z = rng.normal(y,np.sqrt(sig2))
#     RHS.plot(x,z,'.b',alpha= 0.1)
#     RHS.set_ylim(-2, 2)
#
# errors = np.abs(y-groundTruth(x))

LHS.plot(x,groundTruth(x),':g',xData,yData,'ok')
# RHS.plot(xData,yData,'ok')


