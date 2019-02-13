---
layout: post
title: "Segue from Euclidean Gradient Descent to Natural Gradient Descent"
tags: [Gradient descent, Natural gradients]
comments: true
mathjax: true
---

*tldr-* A slight change in SGD formulation, in terms of maximation of local approximation, leads to an interesting general connection to NGD via mirror descent.


Natural Gradient descent has recently been gaining quite a significant amount of attention (and rightly so!), since it was originally proposed by Shun-ichi Amari way back in the late 1990s. Especially the Approximate Bayesian Inference group at RIKEN-AIP, Tokyo, of which I am currently a part of, have [successfully applied](https://emtiyaz.github.io/publications.html) Natural gradients to a range of complex Bayesian models.

In this post, I shall discuss one simple yet interesting segue from gradient descent in Euclidean space to Natural gradient descent. In an [earlier blog post](https://antixk.github.io/blog/nat-grad-exp-fam/) , we discussed the relationship between gradient descent and natural gradient descent for exponential family of distributions. In this post, we shall see a more generic connection between them, leveraging the results of Raskutti et. al [2].

Consider the standard SGD update with respect to the learning parameters $\boldsymbol{\theta}$ at time step $t+1$ in Euclidean space as follows

$$
\begin{align}
\boldsymbol \theta_{t+1} = \boldsymbol \theta_t + \beta_t \hat{\nabla}_{\boldsymbol{\theta}}L(\boldsymbol{\theta}_t)
 \end{align}
$$
Where $\hat{\nabla}_{\boldsymbol{\theta}}L(\boldsymbol{\theta}_t)$ is the gradient of the loss function with respect to the parmaters and $\beta_t$ is the current learning rate. The above SGD udpate can be reformulated as a local approximation maximiation as follows -

$$
\begin{align}
\boldsymbol{\theta}_{t+1} = \underset{\boldsymbol{\theta}\in \Theta}{\mathsf{argmax}} \big\langle \boldsymbol \theta, \hat{\nabla}_{\boldsymbol{\theta}}L(\boldsymbol{\theta}_t) \big \rangle - \frac{1}{2\beta_t}
 \| \boldsymbol{\theta} - \boldsymbol{\theta}_t\|_2^2
 \end{align}
 $$

From a probabilistic perspective, $\boldsymbol \theta$ is the *natural paramter* of the modelling distribution. In other words, the model tries to learn the data using the distribution $q(\boldsymbol{\theta})$, whose natural parameters $\boldsymbol{\theta}$ are learned during training.

#### Proximity Function
In the above equation, the Euclidean norm is actually a *proximity function* that measures the discrepancy of the loss function and its linear approximation with respect to the local geometry. This is quadratic term is exact for convex landscapes while provides a somewhat decent approximation for others. 

From a different perspective, the proximity function can also be viewed as a prior belief about the loss landscape. Second order methods like that of Newton's, do away with linear approximation and  directly employ the Hessian to obtain the local quadratic curvature (which is still an approximation). Natural gradients, on the other hand, use the Riemannian curvature tensor to capture the exact local geometry of the landscape.

### The Connection
We rewrite the above maximization problem in terms of the expectation(mean) parameters of the distribution $q$, and use the KL divergence for the proximity function, to get the following mirror descent formulation.

$$
\begin{align}
\boldsymbol{\mu}_{t+1} = \underset{\boldsymbol{\mu}\in M}{\mathsf{argmax}} \big\langle \boldsymbol \mu, \hat{\nabla}_{\boldsymbol{\mu}}L(\boldsymbol{\theta}_t) \big \rangle - \frac{1}{\beta_t}
 KL(q_m(\boldsymbol{\theta})\|q_m(\boldsymbol \theta_t))
  \end{align}
 $$

Instead of performing the parameter update on the natural parameter space, we are updating its *dual* - the expectation parameters. The now, interesting the connection is that, the above mirror descent update on the mean parameters, is equivalent to performing natural gradient update on the natural parameters. Mathematically, we write this as

$$
\begin{align}
\boldsymbol{\mu}_{t+1} &= \underset{\boldsymbol{\mu}\in M}{\mathsf{argmax}} \big\langle \boldsymbol \mu, \hat{\nabla}_{\boldsymbol{\mu}}L(\boldsymbol{\theta}_t) \big \rangle - \frac{1}{\beta_t}
 KL(q_m(\boldsymbol{\theta})\|q_m(\boldsymbol \theta_t)) \\
\boldsymbol{\theta}_{t+1} &= \underset{\boldsymbol{\theta}\in \Theta}{\mathsf{argmax}} \big\langle \boldsymbol \theta, \hat{\nabla}_{\boldsymbol{\theta}}L(\boldsymbol{\theta}_t) \big \rangle - \frac{1}{2\beta_t}
 \| \boldsymbol{\theta} - \boldsymbol{\theta}_t\|_2^2
 \end{align}
 $$

#### Why Mirror Descent?
Mirror descent is a framework that accounts for the geometry of the optimization landscape. It is a generalized framework that incorporates almost all optimization algorithms and over high dimensions. For example, in high dimensions, the local linear or quadratic approximation of the loss surface usually fails. Therefore, it is desirable to employ the actual local geometry of the loss landscape, and mirror descent framework provides an elegant way to exactly that! The second term in the above two maximization formulations (a.k.a proxmitiy function) $-$ the Euclidean norm and the KL divergence $-$ represents the movement of the parameters taking into account the geometry of the landscape. For a more detailed description of the mirror descent method, refer [this document](http://www.princeton.edu/~yc5/ele538_optimization/lectures/mirror_descent.pdf).


SGD update given above, therefore, works in the natural parameter space. Note that the Euclidean norm in the second term, indicating that the descent happens in the Euclidean space.

### Summing Up

|                      | **Gradient Descent**         | **Natural Gradient Descent**                   |
|--------------------- | ------------------------- | ------------------------------------------ |
| **Geometry**             | Euclidean Geometry        | Statistical Manifold (Riemannian Geometry) |
| **Proximity Function**   | Euclidean Norm            | Bergman Divergence (Ex. KL Divergence)     |
| **Optimization Parameters**  | Natural Parameters $\boldsymbol \theta$ | Mean Parameters $\boldsymbol{\mu}$    |
|======================|===========================|========================= |


**Equivalence** : Natural Gradient Descent in $\boldsymbol{\theta}$  $\Leftrightarrow$ Mirror Descent in $\boldsymbol{\mu}$ ||

Apart from the above beautiful relations, there exists yet another equivalence for exponential family of distributions as -

$$
\begin{align}
\mathbf{F}(\boldsymbol{\theta})^{-1}\nabla_{\boldsymbol{\theta}}L(\boldsymbol{\theta})=\nabla_{\boldsymbol{\mu}}L(\boldsymbol{\mu})
 \end{align}
$$
That is, the natural gradient with respect to $\boldsymbol{\theta}$ is simply the gradient of the loss function with respect to $\boldsymbol{\mu}$.

#### References
[1] Amari, Shun-ichi, and Scott C. Douglas. "Why natural gradient?." ICASSP. Vol. 98. No. 2. 1998.

[2] Raskutti, Garvesh, and Sayan Mukherjee. "The information geometry of mirror descent." IEEE Transactions on Information Theory 61.3 (2015): 1451-1457.

[3] Khan, Mohammad Emtiyaz, et al. "Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam." arXiv preprint arXiv:1806.04854 (2018).


