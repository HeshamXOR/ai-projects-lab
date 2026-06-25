# EXPLAINER -- the statistics, from first principles

This document derives every method in `core/` and explains the numerical
choices. The unifying theme: **all of frequentist A/B testing reduces to two
operations on a distribution -- the CDF (statistic -> probability) and its
inverse (probability -> critical value).** We implement those for the normal and
Student-t distributions ourselves, then build everything on top.

---

## 1. The backbone: from-scratch distributions (`core/distributions.py`)

### 1.1 The error function `erf`

The standard normal CDF is defined through the error function:

```
erf(x) = (2/sqrt(pi)) * integral_0^x e^{-t^2} dt
```

There is no elementary closed form, so we approximate. `_erf_series` implements
the Abramowitz & Stegun 7.1.26 rational form:

```
erf(x) ~= 1 - (a1 t + a2 t^2 + ... + a5 t^5) * e^{-x^2},   t = 1/(1 + p|x|)
```

with sign folding for negative `x`. Maximum absolute error ~1.5e-7 -- well inside
every tolerance we need. The public `erf`/`erfc` default to `math.erf`/`math.erfc`
(machine precision) for the live path while the hand-rolled series is unit-tested
against `math.erf` to prove the from-scratch implementation is correct.

### 1.2 Normal CDF and survival function

```
Phi(x) = 1/2 * (1 + erf(x / sqrt(2))) = 1/2 * erfc(-x / sqrt(2))
```

We use the `erfc` form because it is numerically stable in the tails (avoids
cancellation `1 - (almost 1)`). This gives `Phi(0) = 0.5` exactly and
`Phi(1.96) = 0.97500...`.

### 1.3 Inverse normal CDF (`normal_ppf`)

We need `z` such that `Phi(z) = p` -- e.g. `Phi^{-1}(0.975) = 1.959964`, the
critical value behind every 95% test. There is no closed form, so we use **Peter
Acklam's** piecewise rational approximation (three regions: lower tail, central,
upper tail), which alone achieves relative error < 1.15e-9. We then apply one
**Halley iteration** using our own `Phi` and `phi`:

```
e = Phi(x) - p
u = e * sqrt(2 pi) * exp(x^2 / 2)
x <- x - u / (1 + x*u/2)
```

Halley is cubically convergent, so a single step takes the already-excellent
seed to full double precision. This function powers every critical value and the
entire power/sample-size module.

### 1.4 Student-t CDF via the incomplete beta function

The t-distribution has heavier tails than the normal; for small samples this
matters enormously (`df=1` is the Cauchy distribution). The CDF is **not** a
Gaussian approximation -- it is exact, expressed through the **regularized
incomplete beta function** `I_x(a, b)`:

```
x = df / (df + t^2)
F_t(t; df) = 1 - 0.5 * I_x(df/2, 1/2)   for t >= 0
           =     0.5 * I_x(df/2, 1/2)   for t <  0
```

`I_x(a, b) = B(x; a, b) / B(a, b)` is computed with the classic **Lentz
continued fraction** (`_betacf`), the same algorithm "Numerical Recipes" uses.
We apply the symmetry relation `I_x(a,b) = 1 - I_{1-x}(b,a)` so the continued
fraction is always evaluated where it converges fastest. The log-Beta normalizer
uses `math.lgamma`. Accuracy is ~1e-12 across the parameter ranges we hit, and
the test-suite checks it against textbook t critical values (2.228 at df=10,
12.706 at df=1, etc.) and against the normal limit as `df -> inf`.

---

## 2. Two-proportion z-test (`core/tests_stats.py`)

For a conversion metric, each arm is `Binomial(n, p)`. The difference of sample
proportions `p_b - p_a` is asymptotically normal. Two SEs appear, for two
different purposes:

- **Pooled SE (for the test statistic).** Under H0 the two arms share one rate,
  so we pool: `p_hat = (x_a + x_b)/(n_a + n_b)` and
  `SE = sqrt(p_hat(1-p_hat)(1/n_a + 1/n_b))`. The statistic is
  `z = (p_b - p_a) / SE`, p-value `= 2(1 - Phi(|z|))`.
- **Unpooled SE (for the confidence interval).** Off the null the rates differ,
  so the CI uses `SE = sqrt(p_a(1-p_a)/n_a + p_b(1-p_b)/n_b)` and
  `CI = (p_b - p_a) +/- z_{1-alpha/2} * SE`.

Using the pooled SE for the test and the unpooled SE for the CI is the standard,
correct practice (a CI built from the pooled SE would be inconsistent with the
point estimate off the null). The worked example in the tests
(100/1000 vs 130/1000) gives `z = 2.103`, `p = 0.0355`.

---

## 3. Welch's t-test (`core/tests_stats.py`)

For a continuous metric with possibly unequal variances we use **Welch's t**, not
the pooled Student t (the equal-variance assumption is rarely justified online):

```
se = sqrt(s_a^2/n_a + s_b^2/n_b)
t  = (mean_b - mean_a) / se
```

The degrees of freedom come from the **Welch-Satterthwaite** approximation,
which matches the moments of the linear combination of two chi-squared
variables:

```
df = (s_a^2/n_a + s_b^2/n_b)^2
     / ( (s_a^2/n_a)^2/(n_a - 1) + (s_b^2/n_b)^2/(n_b - 1) )
```

This `df` is generally non-integer. The p-value is `2 * (1 - F_t(|t|; df))` using
our incomplete-beta t-CDF. The CI needs the t critical value `t_{1-alpha/2, df}`;
since the t-CDF has no closed inverse, `_t_critical` **bisects** our own t-CDF
(seeded from the normal quantile and bracket-expanded), converging to 1e-10. The
textbook example (means 20/22, variances 16/25, n 10/12) yields `t = 1.042`,
`df = 19.98`.

---

## 4. Power and sample size (`core/power.py`)

The two-sided sample size to detect an absolute lift from `p` to `q = p + MDE`:

```
n_per_arm = ( z_{1-alpha/2} * sqrt(2 p_bar (1 - p_bar))
              + z_{1-beta}  * sqrt(p(1-p) + q(1-q)) )^2 / (q - p)^2
```

where `p_bar = (p+q)/2` and `z_{1-beta}` is the quantile for the target power.
The first term reflects the null-hypothesis spread (pooled); the second the
alternative spread. Everything flows through `normal_ppf`. The example
(p=0.10, MDE=0.02, alpha=0.05, power=0.80) gives ~3841 per arm, matching standard
calculators.

`power_for_sample_size` inverts the relation -- given a fixed `n`, solve for
`z_beta` and map through `Phi` to report the **achieved power**, which the API
uses to say "you currently have 62% power; keep running". `sample_size_for_mean`
is the continuous-metric analogue:
`n = 2 sigma^2 (z_{1-alpha/2} + z_{1-beta})^2 / MDE^2`.

---

## 5. Sequential testing (`core/sequential.py`)

Repeatedly peeking at a fixed-horizon test inflates the false-positive rate
massively. Two principled remedies are implemented.

### 5.1 Wald's SPRT

We accumulate the log-likelihood ratio of the data under H1 vs H0. For `k`
Bernoulli successes in `n` trials:

```
LLR = k * log(p1/p0) + (n - k) * log((1-p1)/(1-p0))
```

with horizontal decision boundaries derived directly from the target error
rates:

```
A = log( beta / (1 - alpha) )   ->  LLR <= A : accept H0 (futility)
B = log( (1 - beta) / alpha )   ->  LLR >= B : reject H0 (effect detected)
```

Between the boundaries we keep sampling. The SPRT minimizes expected sample size
among tests with the given error rates (Wald-Wolfowitz optimality).

### 5.2 Alpha-spending (group-sequential)

The Lan-DeMets framework spends the total `alpha` budget across looks as a
function of the **information fraction** `t = n_current / n_planned`:

- **Pocock:** `alpha*(t) = alpha * ln(1 + (e-1) t)` -- spends roughly evenly, so
  its z-boundary is nearly constant across looks.
- **O'Brien-Fleming:** `alpha*(t) = 2(1 - Phi(z_{1-alpha/2} / sqrt(t)))` -- spends
  almost nothing early (very hard to stop at the first peek) and approaches the
  full `z_{1-alpha/2} = 1.96` boundary at `t = 1`. This is usually preferred
  because it barely penalizes the final analysis.

At each peek we convert the cumulative spend back into a z critical value via
`normal_ppf` and compare it to the observed z. `group_sequential_decision` returns
**stop-reject** (boundary crossed), **stop-accept** (full information, sub-boundary
z = futility), or **continue**.

---

## 6. CUPED variance reduction (`core/cuped.py`)

CUPED (Deng et al., WSDM 2013) uses a **pre-experiment covariate** `X` -- measured
before assignment, hence independent of treatment -- to remove predictable
variance from the metric `Y`:

```
Y_cuped = Y - theta * (X - E[X]),   theta = Cov(Y, X) / Var(X)
```

`theta` is just the OLS slope of `Y` on `X`. Because `X` predates treatment,
`E[X]` is equal across arms in expectation, so the adjustment is **unbiased**:
`E[Y_cuped] = E[Y]`, and the estimated treatment effect is preserved. The payoff:

```
Var(Y_cuped) = Var(Y) * (1 - rho^2)
```

where `rho = corr(Y, X)`. A covariate correlated 0.8 with the metric removes
`0.64` of the variance -- equivalent to ~2.8x the sample size, for free. We
estimate a **single pooled `theta`** across both arms (the standard recipe, which
avoids leaking the treatment effect into the slope), produce adjusted per-arm
means and variances, and report the achieved variance reduction. The adjusted
summaries feed straight into Welch's t-test, so the whole pipeline composes.

The test-suite constructs `Y = rho*X + sqrt(1-rho^2)*eps` and asserts that the
adjusted variance is strictly below the raw variance in both arms, that the
reduction is ~`rho^2`, and that a known injected treatment effect survives the
adjustment.

---

## 7. Why no SciPy?

Every distribution function here is a few dozen lines of well-understood
numerical analysis. Implementing them ourselves (a) removes a heavy dependency,
(b) makes the statistical machinery fully auditable, and (c) demonstrates that
the "magic" of A/B testing is just the normal and t distributions plus careful
algebra. The test-suite pins each function to textbook constants so correctness
is verifiable without any reference library.
