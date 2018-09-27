---
layout: post
title: "Lipschitz Regularization Methods for WGANs"
tags: [GANs, Deep Learning, code]
comments: true
mathjax: true
---

*TL;DR-* Code snippets for various Lipschitz Regularization methods for WGAN - Gradient Clipping, Gradient Penalty, Spectral Normalization etc. in PyTorch.

Wasserstein Generative Adversarial Networks (WGANs) have attracted a lot of research interests for two main reasons -
1) Noting the fundamental difference between the two classes of probability metrics - $$f-$$divergences and IPMs (Integral Probability Metrics) and correctly opts for the widely used IPM metric - the Wasserstein Distance.
2) Simplifying the computation of Wasserstein distance using the *Kantorovich-Rubinstein Duality* which converts this

$$
\begin{align}
W(p_{data}, q_\theta) = \inf_{\gamma \in \pi(p_{data}, q_\theta)} \mathbb{E}_{x,y \sim \gamma}[\|x-y\|]
\end{align}
$$

   to this

$$
\begin{align}
W(p_{data}, q_\theta) = \sup_{\|f\|_{L \leq 1}} \mathbb{E}_{x \sim p_{data}}[f(x)] - \mathbb{E}_{y \sim q_\theta}[f(y)]
\end{align}
$$

Essentially, we convert an intractable infinimum to a supremum that can be computed. For a complete derivation of the above duality, refer this [awesome post](https://vincentherrmann.github.io/blog/wasserstein/).

The caveat under concern is that how to make the function $$f$$ ,which is the discriminator in case of WGANs, $$1-$$ Lipschitz?. The following are some of the neat ideas for regularizing *any* Discriminator network to $$1-$$ Lipschitz.

### Gradient Clipping
Note that the fundamental idea behind $$1-$$ Lipschitz functions it that they have finite bound on their gradients (of $$1$$ in this case) in the given interval. Therefore to achieve this, one solution is to clip the weights of the Discriminator - which controls how the discriminator behaves - such that the gradient value never exceeds $$1$$. This is stright-forward to implement in Pytorch using the `clamp` function.

```python
def gradient_clipping(netD, clip_val=0.01):

  for p in netD.parameters():
    p.data.clamp(-clip_val, clip_val)
```
However, the above clipping procedure is not a good strategy to enforce the Lipschitz constraint. This is because the clip operation reduces the space of discriminators that maximize the WGAN objective and in the wost cases, the optimal discriminator may not even be obtained [5].

### Gradient Penalty
Gradient Penalty is another solution to regularize the gradients of the discriminator network using the Lagrange Multiplier approach. This approach viwes the Lipschitz condition as a constraint to the existing training objective of the WGAN and therefore adds a penalty if the gradient is larger than $$1$$.

```python
def gradient_penalty(netD, real_data, fake_data, lambda_val=10):
    '''
    Gradient Penalty in WGAN computer as follows -.
    1) Taking a point along the manifold connecting
    the real and fake data points and computing the gradient at that point.
    2) Computing the MSE of the gradient from the value 1.
    ------------------------
    :param netD: Discriminator Network
    :param real_data: Real data - Variable
    :param fake_data: Generated data - Variable
    :param lambda_val: coefficient for the gradient Penalty
    :return: Gradient Penalty
    '''
    #Interpolate Between Real and Fake data
    shape = [real_data.size(0)] + [1] * (real_data.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = real_data + alpha * (fake_data - real_data)

    # Compute Gradient Penalty
    z = Variable(z, requires_grad=True).cuda()
    disc_z = netD(z)

    gradients = grad(outputs=disc_interpolates, inputs=z,
                              grad_outputs=torch.ones(disc_z.size()).cuda(),
                              create_graph=True)[0].view(z.size(0), -1)

    gradient_penalty = ((gradients.norm(p=2, dim=1) - 1) ** 2).mean() * lambda_val
    return gradient_penalty
```

### Consistency Regularization
Consistency Regularization is an improvement to the Gradient penalty term. Note that the gradient penalty term takes the gradient only on the manifold connecting the real and fake data points, proving continuity. The consistency Regularization (CT) complements the gradient penalty term by emphasizing more on the region around the real data points. This can be done by adding some noise/perturbations to the input $$x + \delta$$. However, the authors in the paper[2] suggest to use the stochasticity of the discriminator network (stochastic dropout) to yield better results.

```python
def consistency_regularization(netD, real_data, lambda_val=2, M_val = 0.0):
  '''
  Consistency regularization complements the gradient penalty by biasing towards
  the real-data along the manifold connecting the real and fake data.
  ---------------------
  :param netD: Discriminator network that returns the output of the last layer
               and the pen-ultimate layer.
  :param real_data: Real data - Variable
  :param lambda_val: coefficient for the consistency_regularization term
  :param M_val: constant offset M ~ [0, 0.2]
  :return: consistency regularization term
  '''
  dx1, dx1_ = netD(real_data)
  dx2, dx2_ = netD(real_data) # Different from dx1 because of stochastic dropout
  CT = (dx1 - dx2)**2 + 0.1*(dx1_ - dx2_)**2
  cons_reg = torch.max(torch.zeros(CT.size()), lambda_val*CT - M_val).mean()
  return cons_reg
```
Note that in the above code, the discriminator network returns both the output of the last layer as well the pen-ultimate layer. This is again, because the authors[2] remark that regularizing the pen-ultimate layer improves the results further.

### Spectral Normalization
Spectral Normalization takes a vastly different approach and tackles the problem head-on. The idea is based on the simple relation - The Lipschitz constant $$M$$ of a transformation function is equivalent to its *spectral norm*. Therefore, the weights of the discriminator network can simply be scaled down by its spectral norm to regularize its Lipschitz constant to $$1$$.

Note that the spectral norm of a matrix is its largest singular value, which can be easily found using the Power Iteration method, shown in the following code snippet.

```python
def _L2Normalize(v, eps=1e-12):
    return v/(torch.norm(v) + eps)

def spectral_norm(W, u=None, Num_iter=100):
    '''
    Spectral Norm of a Matrix is its maximum singular value.
    This function employs the Power iteration procedure to
    compute the maximum singular value.
    ---------------------
    :param W: Input(weight) matrix - autograd.variable
    :param u: Some initial random vector - FloatTensor
    :param Num_iter: Number of Power Iterations
    :return: Spectral Norm of W, orthogonal vector _u
    '''
    if not Num_iter >= 1:
        raise ValueError("Power iteration must be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0,1).cuda()
    # Power iteration
    for _ in range(Num_iter):
        v = _L2Normalize(torch.matmul(u, W.data))
        u = _L2Normalize(torch.matmul(v, torch.transpose(W.data,0, 1)))
    sigma = torch.sum(F.linear(u, torch.transpose(W.data, 0,1)) * v)
    return sigma, u
```

#### References
[1] Arjovsky, Martin, Soumith Chintala, and Léon Bottou. "Wasserstein generative adversarial networks." International Conference on Machine Learning. 2017.

[2] Gulrajani, Ishaan, et al. "Improved training of wasserstein gans." Advances in Neural Information Processing Systems. 2017.

[3] Wei, Xiang, et al. "Improving the Improved Training of Wasserstein GANs: A Consistency Term and Its Dual Effect." arXiv preprint arXiv:1803.01541 (2018).

[4] Miyato, Takeru, et al. "Spectral normalization for generative adversarial networks." arXiv preprint arXiv:1802.05957 (2018).

[5] Petzka, Henning, Asja Fischer, and Denis Lukovnicov. "On the regularization of Wasserstein GANs." arXiv preprint arXiv:1709.08894 (2017).
