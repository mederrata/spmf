We want to characterize the scale of elements in the matrix product $\mathbf{X}=\mathbf{U}\mathbf{V},$ where $u_{ij} = f_u(a_{ij})$ and $v_{jk} = f_v(b_{jk})$ and $a_{ij}\sim^{iid} \mathcal{N}(\mu_a, \sigma_a)$ and $b_{jk}\sim^{iid} \mathcal{N}(\mu_b, \sigma_b)$, and $f_u, f_b$ are invertible smooth functions.

Each matrix element takes the form of the inner product
$$
x_{ik} = \sum_j u_{ij}v_{jk}.
$$

By statistical independence,

$$
\mathbf{E}(x_{ik}) = J\mathbf{E}[f(a)]\mathbf{E}[f(b)].
$$

where $a\sim \mathcal{N}(\mu_a, \sigma_a)$ and $b\sim \mathcal{N}(\mu_b, \sigma_b)$. Also by statistical independence, 

$$
\textrm{Var}(x_{ik}) = J\textrm{Var}\left[f_u(a)f_v(b) \right]
$$

where

$$
\textrm{Var}\left[f_u(a)f_v(b) \right]=\textrm{Var}[f_u(a)]\textrm{Var}[f_v(b)] + \textrm{Var}[f_u(a)]\mathbb{E}[f_v(b)]^2 + \textrm{Var}[f_v(b)]\mathbb{E}[f_u(a)]^2 
$$