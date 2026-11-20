# Sampled Embedding Covariance Regularizer

## Scope and terminology

The optimizer applies this regularizer to sampled rows of the tied
`sensory.embedding.weight` table. In this model that table is also used by the
output projection. It does not regularize hidden states or label semantic
contradictions between tokens.

The asymmetric objective is a centered pairwise similarity penalty. It is not
the same quantity as feature-channel covariance. “Covariance” in the feature
mode below refers to channel covariance across sampled embedding rows.

## Asymmetric mode

For a sampled matrix of embedding rows `E` with shape `M x D`, center each
feature across sampled tokens:

`X = E - mean(E, axis=0)`

Define the symmetric pairwise similarity matrix `K = X X^T / D`, set its
diagonal to zero, and retain only positive entries `A = max(K, 0)` off-diagonal.
The sampled loss is:

`L_asym = lambda / (M (M - 1)) * sum(A ** 2)`

Its gradient with respect to the centered rows is:

`dL/dX = 4 lambda / (D M (M - 1)) * A X`

The gradient is projected through the centering operation and scattered into
the sampled embedding rows. The implementation computes this gradient directly
and adds it before SpheroidOptimizer's existing clipping and update rules; it
does not retain an autograd graph for the regularizer.

This objective discourages positive similarities in the sample. It does not
require all token pairs to become negative: for vocabularies larger than the
embedding dimension, that goal is generally infeasible. It also does not
identify which pairs are contradictory or mutually exclusive.

## Full decorrelation mode

For centered rows `X`, compute `Sigma = X^T X / (M - 1)`, remove its diagonal,
and use:

`L_full = lambda / (2 D) * ||Sigma - diag(Sigma)||_F^2`

The analytic gradient is:

`dL/dX = 2 lambda / (D (M - 1)) * X (Sigma - diag(Sigma))`

This drives off-diagonal feature covariance toward zero. It does not force
negative covariance and does not by itself guarantee full rank or semantic
independence.

## Runtime defaults

Training exposes `--neg-cov-lambda`, `--neg-cov-mode`, `--neg-cov-subsample`,
and `--neg-cov-interval`. Current defaults are `0.05`, `asymmetric`, `256`,
and `16`. Thus at most 256 vocabulary rows are sampled once every 16 optimizer
updates. The pairwise matrix is `256 x 256`; compute is quadratic in the sample
size, so the earlier claim of negligible or zero overhead is not guaranteed.
Measure throughput and held-out quality on the target accelerator before
increasing the sample size or applying it every step.

## Scientific claims and evaluation

VICReg and Barlow Twins penalize off-diagonal feature covariance/correlation
toward zero; they do not establish that representations should be negatively
correlated. Uniformity objectives encourage dispersion but do not imply that
every token pair should have negative similarity. Biological inhibition is an
analogy, not evidence that this parameter regularizer models GABAergic
mechanisms.

Do not assume a baseline anisotropy value or a target value without measuring
this model and tokenizer. Evaluate the regularizer against an otherwise
identical control using held-out cross-entropy, retrieval and contradiction
tasks, embedding cosine statistics, singular-value effective rank, optimizer
stability, and measured throughput. Report confidence intervals across seeds;
the regularizer alone cannot guarantee improvement in any of these metrics.
