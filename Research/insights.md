# insights

## classic papers Outline:

1. **A Neural Probabilistic Language Model** — Bengio et al. (2003). ([Journal of Machine Learning Research](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)  )
2. **Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification** — He et al. (2015). ([CV Foundation](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
   * Keep variance stable: for ReLU layers, initialize $W\sim\mathcal N(0,\,2/n_{in})$.
3. **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift** — Ioffe & Szegedy (2015). ([arXiv](https://arxiv.org/pdf/1502.03167))



## 1) *A Neural Probabilistic Language Model* (2003)

Replace large sparse n-gram lookup tables by a parametric model that **learns distributed representations (embeddings)** for words and uses an MLP to estimate conditional probabilities

* Use embeddings $d$ typically 50–1024 depending on data; larger $d$ helps but requires regularization.
* For small corpora, tie embeddings or regularize heavily to avoid overfitting.
* Bigram/MLP relation: an MLP with concatenated embeddings subsumes smoothed n-grams — bigrams correspond to using only $n=2$ and linear (or low-capacity) MLP.


## 2) Kamming He init :: *Delving Deep into Rectifiers* (PReLU & MSRA init)

for very deep rectifier (ReLU) networks, use

1. **Parametric ReLU (PReLU)**: $f(x)=\max(0,x)+a\min(0,x)$ where $a$ is learned per channel — gives slightly better fitting than ReLU with negligible cost.
2. **Initialization tuned for ReLU (MSRA / He init)** that preserves forward activation variance across layers, enabling training of deeper nets from scratch.

### MSRA / He initialization (derivation summary)

Keep $\operatorname{Var}[y_l]$ constant across layers for ReLU-like nonlinearity. For a layer $y = W x$ with $W_{ij}$ i.i.d. zero-mean

* For symmetric activations, Xavier/Glorot uses 

  $$
    \operatorname{Var}(W) = \frac{2}{n_\text{in}+n_\text{out}}
  $$
* For ReLU (which halves variance because negative part zeroed), choose

  $$
  \operatorname{Var}(W) = \frac{2}{n_\text{in}}
  $$

so that $\operatorname{Var}[y] \approx \operatorname{Var}[x]$. 

ie, sample $W_{ij}\sim\mathcal N(0,\;2/n_\text{in})$ or uniform with same variance.

* For ReLU/PReLU conv layers use He init: 
$$
    \sigma = \sqrt{\frac{2.0}{fan_{in}}}
$$

### Where it helped

* He et al. achieved **\~4.94% top-5** on ImageNet using PReLU + MSRA init and deeper architectures

## 3) *Batch Normalization* (BN)

For a mini-batch $\mathcal B = \{x^{(i)}\}$ on some layer pre-activation $u$:

1. Compute mean and variance per feature:

   $$
   \mu_{\mathcal B}=\frac{1}{m}\sum_i u^{(i)},\quad \sigma^2_{\mathcal B}=\frac{1}{m}\sum_i (u^{(i)}-\mu_{\mathcal B})^2
   $$
2. Normalize and rescale:

   $$
   \hat{u}^{(i)}=\frac{u^{(i)}-\mu_{\mathcal B}}{\sqrt{\sigma^2_{\mathcal B}+\epsilon}},\qquad y^{(i)}=\gamma\hat{u}^{(i)}+\beta
   $$

   where $\gamma,\beta$ are learned.

###  mathematical intuition

* Reduces shift in distribution of layer inputs across training steps, improving gradient flow and allowing larger learning rates.
* Adds implicit regularization (noise due to batch statistics).
* Stabilizes deeper networks by keeping activations in a normalized regime; the scale parameter $\gamma$ allows restoration of representational power.
* BN often **reduces or removes need for Dropout** in conv nets.
* At inference, use running averages of $\mu,\sigma^2$. Implement carefully for small batch sizes.


