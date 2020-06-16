# Gaussian Discriminant Analysis
_GDA_ is a __generative learning algorithm__, where it learns P(x|y) rather than,
learning the mapping function between features(x) and labels(y) i.e. what discriminative algorithms do. In this type of algorithm, we try to model the
distribution of the features knowing their source(labels) _assuming_ they come
from a _Gaussian distribution_.

## Getting Started
__What is Gaussian Distribution?__<br/>
It is a classic distribution over single scalar random variable _'x'_, parameterized
by mean(_mu_) and standard deviation(_sigma_). Which looks like a typical bell
curve. It's probability density function is as follows:
<p align="center">
<img src="./files/gauss.PNG" width=300>
</p>


__What is Multivariate Gaussian?__<br/>
A multivariate gaussian is a generalization of the gaussian defined over one
dimensional random variable, to multiple random variable at the same time. These
are vector valued random variable rather than univariate random variable.<br/>
It's probability density function is as follows:
<p align="center">
<img src="./files/multigauss.PNG" width=400>
</p>
<br/>

A multivariate gaussian in 2 dimension would look something like this, where the right hand side image shows the contour plot of the gaussian.
<p align="center">
<img src="./files/multigaussplot.png" width=400>
</p>
<br/>

## Maximum Likelihood Estimates
Multivariate gaussian is __parameterized__ by __mean(_mu_)__ which controls the location of the gaussian and __covariance matrix(_sigma_)__ which controls the shape of the gaussian.<br/><br/>
__How to fit the training set?__<br/>
In order to fit these parameters, we need to maximize the joint likelihood. Once
we do this we would have the maximum likelihood estimates for mu and sigma.<br/>
Refer this [link](https://www.youtube.com/watch?v=Dn6b9fCIUpM) for the derivation
of maximizing the joint likelihood.<br/>
<p align="center">
<img src="./files/maxlike.PNG" width=400>
</p>

## Results
In the above code, GDA is used to perform bi-class classification, by modelling
class A and class B separately. Once we fit Multivariate Gaussians to each class
individually, we can get the probability of any new data point from the probability
density function.

#### Scatter plot of dataset
A two dimensional dataset with 100 data points from class A and 100 data points from
class B, where the red points show their mean.
<p align="center">
<img src="./files/scatter.png" width=400>
</p>

#### Contour plot with decision boundary
After fitting Gaussians to both class independently, we can get an approximation
of the decision boundary as show in the contour plot.
<p align="center">
<img src="./files/contour.png" width=400>
</p>

#### 3D visualization of Gaussians
Better visualization of how GDA fits gaussians to the dataset would be as follows.
<br/>

_Note: This plot is not for the provided dataset. This is to get an picture as of
how GDA fits gaussians to distribution._

<p align="center">
<img src="./files/conplot.jpg" width=400>
</p>
