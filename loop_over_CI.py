import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cal_H(x, y):
    #return np.array([[-x, y], [y, x]])
    r = (x**2 + y**2)**0.5
    A = 0.0025
    B = 0.01
    k = 0.8

    if r > 0.0:
        #coef = 1.0
        coef = A * np.exp(-r*r) + B / r * (np.exp(k*r) - 1) / (np.exp(k*r) + 1)
        return coef * np.array([[-x, y], [y, x]])
    else:
        return np.array([[0, 0], [0, 0]])


def show_pes(xarr, yarr, loop=None):
    X, Y = np.meshgrid(xarr, yarr)

    E0 = np.zeros(X.shape)
    E1 = np.zeros(X.shape)

    Nx, Ny = X.shape

    for ix in range(Nx):
        for iy in range(Ny):
            x = X[ix, iy]
            y = Y[ix, iy]
            eva, evt = la.eigh(cal_H(x, y))
            E0[ix, iy] = eva[0]
            E1[ix, iy] = eva[1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, E0)
    ax.plot_surface(X, Y, E1)

    if loop is not None:
        loopx = []
        loopy = []
        loopz = []
        for r in loop:
            loopx.append(r[0])
            loopy.append(r[1])
            loopz.append(0)
        ax.plot(loopx, loopy, loopz)


def get_loop():
    x0, y0 = 0, 0
    N = 200
    r = 1
    dtheta = 2 * np.pi / (N - 1)
    loop = list()
    for i in range(N):
        theta = dtheta * i
        loop.append([x0 + r * np.cos(theta), y0 + r * np.sin(theta)])
    return loop


xarr = np.linspace(-8, 8, 200)
yarr = np.linspace(-8, 8, 200)
loop = get_loop()

show_pes(xarr, yarr, loop)

lastevt = None
phase = np.zeros(2)

Nl = len(loop)
evt00 = np.zeros(Nl)
evt01 = np.zeros(Nl)
evt10 = np.zeros(Nl)
evt11 = np.zeros(Nl)

for i, r in enumerate(loop):
    x, y = r[0], r[1]
    H = cal_H(x, y)
    eva, evt = la.eigh(H)

    if lastevt is not None:
        for k in range(2):
            if np.dot(np.conj(lastevt[:,k]), evt[:,k]) < 0:
                evt[:, k] *= -1

    lastevt = np.copy(evt)

    evt00[i] = evt[0,0]
    evt10[i] = evt[1,0]
    evt01[i] = evt[0,1]
    evt11[i] = evt[1,1]


plt.figure()
plt.plot(np.arange(Nl), evt00, label='evt00')
plt.plot(np.arange(Nl), evt01, label='evt01')
plt.plot(np.arange(Nl), evt10, label='evt10')
plt.plot(np.arange(Nl), evt11, label='evt11')
plt.legend()
plt.show()
