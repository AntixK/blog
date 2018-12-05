# A Cleverer Trick on top of the Reparametrization Trick

*TL;DR -* Implicit differentiation can lead to an efficient computation of the gradient of reparametrized samples.

The famous *reparametrization trick* has been employed in estimating the gradients of samples from probability distributions by replacing an equivalent estimator that is *deterministic* and a *differential transformation* of a simple distribution.  

The paper expounds the requirements of probability distributions on which the reparametrization trick can be used. For the reparametrization trick, the probability distribution, whose sample gradients are required, must satisfy at least one of the following conditions -
- Has location-scale parametrization
- Has a tractable inverse (cumulative distribution functions) CDF
- Can be expressed as a deterministic differential transformation of other distributions satisfying the above two conditions.

### Revisiting the Reparametrization Trick
The reparametrization trick is used mainly in estimating the gradient of an expectation of a differentiable function $f(x)$ with respect to the parameters of the distribution $q(x; \theta)$ such that $x \sim q(x; \theta)$. In other words, the trick can be used to compute $\nabla_\theta E_{q(x;\theta)} [f(x)]$.


Here, $q(x;\theta)$ is come complex distribution that satisfies at least one of the conditions mentioned above.
Note that the main impediment to computing the gradient of the above expression directly is the non-differentiable step of sampling $x$ from $q(x;\theta)$.

Therefore, the trick is to rewrite the argument of the function $f$ as $f(s(x'))$ such that it is independent of the parameters of the distribution. In other words, since the sampling procedure is not differentiable, make the sampling procedure independent of the parameters so that the gradient for the sampling is not required. By re-writing the argument, the parameters get transferred to the function $f$ through $s(x')$.
When written as $f(s(x'))$, the $x'$ here is independent of the parameters of the distribution and hence no gradient of $x$ with respect to $\theta$ is required.

In general, if a sample $x$ can be written as a deterministic differentiable expression $s_{\theta}(x')$ where $x'$ is a sample that is independent of the parameters $\theta$

$$
x = s_{\theta}(x'); \text{ where } x' \sim q(x')\\
\nabla_{\theta} E_{q(x;\theta)} [f(x)] = \nabla_{\theta} E_{q(x;\theta)} [f(s_{\theta}(x'))]
$$
The gradient of the above expression can thus, be computed (using chain-rule) as
$$
\nabla_{\theta} E_{q(x;\theta)} [f(s_{\theta}(x'))] = E_{q(x;\theta)} [\nabla_{\theta}f(s_{\theta}(x'))] = E_{q(x;\theta)} [\nabla_{x}f(s_{\theta}(x')) \nabla_{\theta}s_{\theta}(x')]
$$

Now, if the distribution $q(x; \boldsymbol \theta)$ has a location and scale parameters (like the Gaussian distribution) $\boldsymbol \theta = \{\mu, \sigma\}$, as then $s_\theta(x')$ can be a simple translation and scaling of the form
$$
x = s_{\boldsymbol \theta}(x') = (x' - \mu)/\sigma; \text{ where } x' \sim q(0,1)
$$

If the distribution $q(x; \theta)$  has a tractable inverse CDF $Q^{-1}$, then $x$ can be written as
$$
x = s_{\theta}(x') = Q^{-1}(x'); \text{ where } x' \sim U[0,1]
$$

It is also possible to use both of the above transformations in tandem, justifying the conditions presented above.

However, distributions like Gamma, Beta, Dirichlet distributions or even mixture distributions do not satisfy the above conditions and thus, the reparametrization trick cannot be used. Other techniques, addressing this limitation include approximating the intractable inverse CDF or using *score function* (gradient of the log likelihood). However, these produce gradients with relatively large variance. Large variance in such estimates affect the convergence of the training algorithm, and therefore, further variance-reduction techniques (like that of control-variates) are required. Often, the variance reduction techniques are problem-specific and cannot be used for a wide range of models.

### Implicit Reparametrization
This paper proposes a clever technique for producing low-variance gradients using the reparametrization trick, that is applicable over a *large range* of probability distributions. Firstly, the difficulty arises from computing the gradient of the expression $s_{\theta}(x')$. For distributions like Gamma, the expression $s_{\theta}(x')$ usually follows their inverse CDF which is intractable. Therefore, computing the gradient becomes a huge problem. The task now is to find an efficient way to compute the gradient of the expression even for intractable $s_{\theta}$.

The key insight here is that the parameter-independent sample $x'$ can be written as

$$
x' = s_\theta^{-1}(x)
$$
Now, we can apply *implicit differentiation* technique to the above expression as follows-

$$
\nabla s_\theta^{-1}(x) = \nabla x'\\
\nabla_{x}s_{\theta}^{-1}(x)\nabla_{\theta}x + \nabla_{\theta}s_{\theta}^{-1}(x)\nabla_{x}x = 0\\
\nabla_{\theta}x = -(\nabla_{x}s_{\theta}^{-1}(x))^{-1} \nabla_{\theta}s_{\theta}^{-1}(x)
$$
(Note that $\nabla$ represents total gradient and $\nabla_{\theta}$ represents gradient with respect to $\theta$.)
Therefore, through implicit differentiation, it is possible to find the gradient of the reparametrized samples $x$.

Now, observe that $\nabla_{\theta}x$ is simply $\nabla_{\theta}s_{\theta}(x')$. Since implicit differentiation yields the same result as that of the usual differentiation, the overall results for easier distributions like Gaussian are identical to the usual procedure. Furthermore, note that the above expression is only in terms of $s_{\theta}^{-1}$ which is essentially the CDF of complicated distributions like Gamma distribution. In such cases, numeric differentiation can be used to find the gradients.

In conclusion, using implicit differentiation, a generic method for finding the gradient of the reparametrized expression $s_{\theta}(x')$ can be determined. In cases where the CDF is intractable, the gradient can be directly found using numeric differentiation, as opposed to inverting the CDF and then computing the gradient in the usual reparametrization trick.

#### References
[1] Figurnov, Michael, Shakir Mohamed, and Andriy Mnih. "Implicit Reparameterization Gradients." arXiv preprint arXiv:1805.08498 (2018).
