<h1>Maximum Entropy</h1>

# Assumption

Maximum entropy model with the highest `entropy` is the best model among all of the models.

# Definition

For training data $T=\{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\}$, 

and features $F = \{f_i(x, y)\}, i=1, 2, ..., n_F$.

> $f(x, y) = 1$, if $(x, y)$ satisfies some facts, $0$ otherwise.

Maximum entropy is a discriminative model, $P = P(Y|X)$.

## Conditioned entropy

> $H(Y|X) = \sum_{x}P(x)H(Y|X=x) = -\sum_x P(x) \sum_y P(y|x)\log(P(y|x))$

> $=-\sum_{x, y}P(x,y)\log(P(y|x))$

## Empirical statistics

In the training data with total number of data $N$, 
$\tilde{P}(x, y)$ and $\tilde{P}(x)$ can be computed by count,

> $\tilde{P}(X=x, Y=y) = \frac{count(X=x, Y=y)}{N}$

> $\tilde{P}(X=x) = \frac{count(X=x)}{N}$


## Maximum Entropy 

> $H(P) = H(Y|X) = -\sum_{x, y} P(x, y) \log P(y|x)$

> $= -\sum_{x, y} \tilde{P}(x) P(y|x) \log(y|x)$

## Subjects

Conditions, let the expectation of feature functions $f(x, y)$ be 
equal to the empircal expecation.

> $\mathbb{E}_P(f) = \sum_{x, y} P(x, y) f(x, y) = \sum_{x, y} \tilde{P}(x) P(y|x) f(x, y)$

> $\mathbb{E}_{\tilde{P}}(f) = \sum_{x, y} \tilde{P} (x, y) f(x, y)$

Thus, 

> $s.t. \mathbb{E}_p(f_i) = \mathbb{E}_{\tilde{P}}(f_i), i=1, 2, ..., n_F$

And the sum of probability of output values is 1, 

> $s.t. \sum_y P(y|x) = 1$

## Objective

> $max H(P) = H(Y|X) = -\sum_{x, y} \tilde{P}(x) P(y|x) \log P(y|x)$

> $s.t. \mathbb{E}_p(f_i) = \mathbb{E}_{\tilde{P}}(f_i), i=1, 2, ..., n_F$

> $\sum_y P(y|x) = 1$

By conventions, using the $min$ way, 

> $min -H(P) = \sum_{x, y} \tilde{P}(x) P(y|x) \log P(y|x)$

> $s.t. \mathbb{E}_p(f_i) = \mathbb{E}_{\tilde{P}}(f_i), i=1, 2, ..., n_F$

> $\sum_y P(y|x) = 1$

# Parameter estimation

Use Lagrange method, let $w_0, w_1, ..., w_{n_F}$ be the Larange multipliers, 
the Lagrange function will be, 

> $L(P, w) = -H(P) + w_0 (1 - \sum_y P(y|x)) + \sum_{i=1}^{n_F} w_i 
(\mathbb{E}_p(f_i) - \mathbb{E}_{\tilde{P}}(f_i))$

## Dual problem

### Primal problem 

For a typical optimization problem, 

> $min f_0(x)$

> $s.t.  f_i(x) \le 0, i=1, 2, ..., n_F$

> $g_j(x) = 0, j = 1, 2, ..., n_G$

Lagrange function is, 

> $L(x, \lambda, \gamma)=f_0(x) + \sum_i \lambda_i f_i(x) + \sum_j \gamma_j g_j(x)$

> $s.t. \lambda_i \ge 0, i = 1, 2, ..., n_F$

If $x$ violates the constraints, $\max_{\lambda, \gamma; \lambda \ge 0} L(x, \lambda, \gamma)= + \infty$, otherwise $f_0(x)$.

The `primal problem` is $\min_{x} \max_{\lambda, \gamma; \lambda \ge 0} L(x, \lambda, \gamma)$, which get the mininum of $f_0(x)$ subjects to the constraints.

### Dual function


Let $g(\lambda, \gamma)$ be the infimum of the Lagrange function.

> $g(\lambda, \gamma) = \inf_x L(x, \lambda, \gamma)$


### Dual problem

`Dual problem` is, 

> $\max g(\lambda, \gamma)$

> $=\max_{\lambda, \gamma; \lambda \ge 0} \min_x L(x, \lambda, \gamma)$

> $\le \min_x \max_{\lambda, \gamma; \lambda \ge 0} L(x, \lambda, \gamma)$

#### Weak duality

If dual problem has optimal solution $d^*$, primal problem has optimal solution $p^*$, 

> $d^* \le p^*$.

For all optimization problems, even if the primal problem is not convex.


#### Strong duality

If Stater or KKT conditions are statisfied,

> $d^* = p^*$

##### Stater conditions

$f_0(x)$ and $f_i(x)$ are convex, $h_j(x)$ are affine, 
and the inequality are strictly statisified.

> $\forall \theta$, $C$ is affine if $x_1 \in C$ and $x_2 \in C$, then $\theta * x_1 + (1-\theta) * x_2 \in C$.

> $\forall 0 \le \theta \le 1$, $C$ is convex if $x_1 \in C$ and $x_2 \in C$, then $\theta * x_1 + (1-\theta) * x_2 \in C$.

> $f(x) = Ax + b: R^{k} \rightarrow R^{m}$ is affine, $x \in R^{k}, A \in R^{m*k}, b \in R^m$.

> If $dom f$ is convex, $f: R^n \rightarrow R$ is convex, if $\forall 0 \le \theta \le 1$, $x_1, x_2 \in dom f$, 
$f(\theta * x_1 + (1 - \theta) * x_2) \le \theta * f(x_1) + (1 - \theta) * f(x_2)$.

##### KKT (Karush-Kuhn-Tucker) conditions 

If $f_0(x)$ and $f_i(x)$ are convex, $h_j(x)$ are affine, the inequality of $f_i(x)$ are strictly satisfied ($\exists x$, $\forall i$, $f_i(x) < 0$), there exists $x^*, \lambda^*, \gamma^*$ where $x^*$ are the solution of primal problem and $(\lambda^*, \gamma^*)$ are the solution of dual problem.


$<=>$ (sufficient and necessary)

> $\nabla_x L(x^*, \lambda^*, \gamma^*) = 0$

> $\lambda_i^* f_i(x^*) = 0$

> $f_i(x^*) \le 0$

> $\lambda_i^* \ge 0$

> $h_j(x^*) = 0$

## Maximum entropy prameters estimation 

The Lagrange function of maximum entropy is

> $L(P, w) = -H(P) + w_0 (1 - \sum_y P(y|x)) + \sum_{i=1}^{n_F} w_i 
(\mathbb{E}_p(f_i) - \mathbb{E}_{\tilde{P}}(f_i))$

> $= \sum_{x, y} \tilde{P}(x) P(y|x) \log P(y|x)$

> $+ w_0 (1 - \sum_y P(y|x))$

> $+ \sum_i w_i (\sum_{x, y} \tilde{P}(x)P(y|x) f_i(x, y) - \sum_{x, y} \tilde{P}(x, y) f_i(x, y))$

The primal problem is, 

$\min_{P(Y|X)} \max_{w} L(P, w)$

The dual problem is, 

$\max_w \min_{P(Y|X)} L(P, w)$

$-H(P)$ is convex w.r.t $P(Y|X)$, $(1 - \sum_y P(y|x))$ and $w_i (\mathbb{E}_p(f_i) - \mathbb{E}_{\tilde{p}}(f_i))$ are affine w.r.t $P(Y|X)$.

According to the `Stater conditions`, the optimal solution of dual problem is equal to optimal solution of the primal problem.

Focus on the dual problem, 

> $\max_w \min_{P(Y|X)} L(P, w)$

solve $\min_{P(Y|X) L(P, w)}$ first.

> $L(P, w) = \sum_{x, y} \tilde{P}(x) P(y|x) \log P(y|x)$

> $+ w_0 (1 - \sum_y P(y|x))$

> $+ \sum_i w_i (\sum_{x, y} \tilde{P}(x)P(y|x) f_i(x, y) - \sum_{x, y} \tilde{P}(x, y) f_i(x, y))$

> $\frac{\partial{L(P, w)}}{\partial{P(y|x)}}$
> $= \sum_{x, y} \tilde{P}(x) (\log P(y|x) + 1)$

> $- w_0 \sum_y$ ==> $- \sum_{x} \tilde{P}(x) \sum_y w_0 = -\sum_{x, y}\tilde{P}(x)w_0$

> $+ \sum_i w_i \sum_{x, y} \tilde{P}(x)f_i(x, y)$

> $=\sum_{x, y}\tilde{P}(x)(logP(y|x) + 1 - w_0 + \sum_i w_i f_i(x, y))$

Let the partial function equals $0$, for $\tilde{P}(x) > 0$, 

> $P(y|x) = \exp(w_0 - 1 - \sum_i w_i f_i(x, y))$

> $=\frac{\exp(-\sum_i w_i f_i(x, y))}{\exp(1 - w_0)}$

> $=\frac{\exp(-\sum_i w_i f_i(x, y))}{Z_w(x)}$

Consider the constraints of 

> $\sum_y P(y|x) = 1$

> $\sum_y \frac{\exp(-\sum_i w_i f_i(x, y))}{Z_w(x)} = 1$

Thus, the normalized factor is,  

> $Z_w(x) = \sum_y \exp(-\sum_i w_i f_i(x, y))$

The optimal solution of $w^*$ is, 

> $w^* = \arg\max_w P(y|x)$

then the optimal solution of the primal problem $P(Y|X)$ is obtained.

## IIS

Use the result of IIS (Lihang' Book 2nd, p106) directly, 

> $\sigma_i = \frac{1}{M} \log \frac{ \mathbb{E}_{\tilde{P}}(f_i)}
{\mathbb{E}_P(f_i)}$

if $M = \sum_i f_i(x, y)$ is constant.

Then, the `update rule` is,  

$w_i = w_i + \sigma_i$


> $\mathbb{E}_{\tilde{P}}(f_i) = \sum_{x, y}\tilde{P}(x, y)f_i(x, y)$

> $\mathbb{E}_P (f_i) = \sum_{x, y}\tilde{P}(x)P(y|x)f_i(x, y)$

> $P(y|x) =\frac{\exp(-\sum_i w_i f_i(x, y))}{Z_w(x)}$

> $Z_w(x) = \sum_y \exp(-\sum_i w_i f_i(x, y))$

# Decoding

For the Chinese segmentation problem, given output sequences $O=(o_1, o_2, ..., o_\tau)$, 

the hidden states $I=(i_1, i_2, ..., i_\tau)$ is to be estimated. 

The transist probability $P(h_t|h_{t-1}, o_t)$ is learned.

Let $\sigma_t(s_t)$ be maximum probability among all of the paths with the $t$-th hidden state is $s_t$.

> $\sigma_{t+1}(s') = \max_{s} \sigma_t(s)P(s'|s, o_{t+1})$

> $\sigma_1(s) = P(h_1|h_0, o_t) = 1$
