import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from numpy import *
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import pylab as pl
from numpy.random import uniform, seed
from matplotlib import cm
from scipy.interpolate import griddata

def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    #print(size)
    assert (size == len(mu) and (size, size) == sigma.shape), "dims of input do not match"
    if size == len(mu) and (size, size) == sigma.shape:
        det = linalg.det(sigma)
        assert det!=0, "covariance matrix cannot be singular"

        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = matrix(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result

def get_true(y):
    true = []
    for i in y:
        if(i=="Alaska"):
            true.append(1)
        else:
            true.append(0)
    return true

def seperate_data(x, y):
    alaska = []
    canada = []
    for i,val in enumerate(y):
        if(val=='Alaska'):
            alaska.append(x[i])
        else:
            canada.append(x[i])
    return np.array(alaska), np.array(canada)


def min_max_normalize(x):
    global x1_min, x2_min, x1_max, x2_max
    x1_max = np.max(x[:,0])
    x1_min = np.min(x[:,0])

    x2_max = np.max(x[:,1])
    x2_min = np.min(x[:,1])
    x[:,0] = (x[:,0]-x1_min)/(x1_max-x1_min)
    x[:,1] = (x[:,1]-x2_min)/(x2_max-x2_min)

    return x

def compute_mu(x_a, x_c):
    mu1 = [np.sum(x_a[:,0])/x_a.shape[0], np.sum(x_a[:,1])/x_a.shape[0]]
    mu2 = [np.sum(x_c[:,0])/x_c.shape[0], np.sum(x_c[:,1])/x_c.shape[0]]

    return np.array(mu1), np.array(mu2)

def compute_covar(x_a, mu1):
    sub = x_a - mu1
    covar = np.zeros((2,2))
    for i in sub:
        dot = np.dot(i.reshape(2,1), i.reshape(2,1).T)
        covar = covar + dot
    return covar/x_a.shape[0]

def descion_boundary(mu1, covar_a, mu2, covar_c):
    X1, X2 = np.mgrid[-3:3:100j, -3:3:100j]
    x1_ravel = X1.ravel()
    x2_ravel = X2.ravel()
    rav_data = []
    for rav1, rav2 in zip(x1_ravel,x2_ravel):
        rav_data.append([rav1, rav2])


    dif = []
    for every in rav_data:
        p_a = norm_pdf_multivariate(every, np.squeeze(mu1), matrix(covar_a))
        p_c = norm_pdf_multivariate(every, np.squeeze(mu2), matrix(covar_c))
        dif.append([p_a-p_c])

    dif = np.array(dif)
    dif = dif.reshape(X1.shape)

    return X1, X2, dif

def normalize(fd_list):
    mean = np.mean(fd_list, axis=0)
    sd = np.std(fd_list, axis=0)

    normal_fd = (fd_list - mean)/sd
    return normal_fd

def gauss(x,y,Sigma,mu):
    X=np.vstack((x,y)).T
    mat_multi=np.dot((X-mu[None,...]).dot(np.linalg.inv(Sigma)),(X-mu[None,...]).T)
    return  np.diag(np.exp(-1*(mat_multi)))

def plot_countour(x,y,z):
    # define grid.
    xi = np.linspace(-2.1, 2.1, 100)
    yi = np.linspace(-2.1, 2.1, 100)
    ## grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    # contour the gridded data, plotting dots at the randomly spaced data points.
    CS = plt.contour(xi,yi,zi,len(levels),linewidths=0.5,colors='k', levels=levels)
    #CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
    CS = plt.contourf(xi,yi,zi,len(levels),cmap=cm.Greys_r, levels=levels)
    # plot data points.
    # plt.scatter(x, y, marker='o', c='b', s=5)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title('griddata test (%d points)' % npts)






x = np.loadtxt("input.dat")
y = pd.read_csv("label.dat")
y = np.array(y)
y = np.insert(y, 0, ['Alaska'], axis=0)


#sigma normalization
x = normalize(x)

x_a, x_c = seperate_data(x, y)

# maximum likelihood estimates
mu1, mu2 = compute_mu(x_a, x_c)
mu1 = mu1.reshape(1,2)
mu2 = mu2.reshape(1,2)

print("mu1(Alaska):", np.squeeze(mu1))
print("mu2(canada):", np.squeeze(mu2))

covar_a = compute_covar(x_a, mu1)
covar_c = compute_covar(x_c, mu2)


print("\ncovariance matrix(Alaska):\n",covar_a)
print("\ncovariance matrix(canada):\n",covar_c)


plt.scatter(x_a[:,0], x_a[:,1], marker='o', label='alaska')
plt.scatter(x_c[:,0], x_c[:,1], marker='x', label='canada')
plt.scatter(mu1[0,0], mu1[0,1], c='red')
plt.scatter(mu2[0,0], mu2[0,1], c='red')


plt.xlabel('x1')
plt.ylabel('x2')


#sample datapoint (0.8, 0.2)
p_alaska = norm_pdf_multivariate([0.8, 0.2], np.squeeze(mu1), matrix(covar_a))  #how probable is it that the data point
p_canada = norm_pdf_multivariate([0.8, 0.2], np.squeeze(mu2), matrix(covar_c))  #comes from these to distributions

#their respective probabilities
print("\nprob of data point[0.8, 0.2] belongs to class alaska:", p_alaska)
print("\nprob of data point[0.8, 0.2] belongs to class canada:", p_canada)

#plt.show()

true = get_true(y)

pred = []
for each in x:
    p_a = norm_pdf_multivariate(each, np.squeeze(mu1), matrix(covar_a))
    p_c = norm_pdf_multivariate(each, np.squeeze(mu2), matrix(covar_c))
    if(p_a>=p_c):
        pred.append(1)
    else:
        pred.append(0)


tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
f1 = f1_score(true, pred)




acc = accuracy_score(true, pred)
print("\nf1 score:",f1)
print("accuracy:",acc)


#plotting approximation for the descision boundary
X1, X2, dif = descion_boundary(mu1, covar_a, mu2, covar_c)
pl.scatter(x_a[:,0], x_a[:,1], marker='o', label='alaska')
pl.scatter(x_c[:,0], x_c[:,1], marker='x', label='canada')
pl.contour(X1, X2, dif, levels=[0])
#pl.show()


#Plotting the contours of the gaussians
seed(1234)
npts = 1000
x = uniform(-2, 2, npts)
y = uniform(-2, 2, npts)
z = gauss(x, y, Sigma=covar_a, mu=np.squeeze(mu1))
zz = gauss(x, y, Sigma=covar_c, mu=np.squeeze(mu2))
plot_countour(x, y, z)
plot_countour(x, y, zz)
plt.colorbar() # draw colorbar
plt.show()

