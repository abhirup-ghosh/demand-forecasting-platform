# Methodology

This document explains the reasoning behind the platform: what each part does, the theory
behind it, and why it was chosen. It grows one section at a time as the project develops.

| Section | Status |
|---|---|
| [1. Forecasting models](#1-forecasting-models) | Design written; implementation in progress (PLAN.md P0.6a–e) |
| Backtesting & evaluation | Planned — the fold design exists (P0.5); metrics come with P0.7 |
| Feature engineering | Planned — implemented in P0.4, write-up to follow |

Related documents: [`eda-findings.md`](./eda-findings.md) (the data facts this design responds
to), [`../PLAN.md`](../PLAN.md) (the build plan). Backtest results will go in
`model-evaluation.md` (P0.15). **Nothing in this document reports model results yet.** Where it
says one approach should do better than another, that is a stated expectation to be tested,
not a finding.

---

## 1. Forecasting models

### 1.1 The problem these models solve

- **What we forecast:** daily unit sales for **1782 series** (54 stores × 33 product families).
- **How far ahead:** **28 days** (the horizon *h* = 28), from any cut-off date *T*.
- **What each forecast includes:** a **point forecast** (the single best guess) and an **80%
  prediction interval** (a range the actual value should fall inside 80% of the time).

The data shapes which methods can work (all numbers are from `eda-findings.md`):

- **Strong multiple seasonality.** The day of the week matters a lot (Sunday sells 1.29× the
  average day, Thursday 0.79×), and so does the time of year (December 1.26×).
- **Heterogeneous scale.** Stores differ about 10× in volume, and product families differ by
  orders of magnitude.
- **Zero inflation.** 31% of rows are zero. That includes 12% that are genuine intermittent
  demand, i.e. items that sell only on some days.
- **Useful outside information (covariates)** that is known ahead of time: holidays, and
  promotions, which the retailer plans in advance.
- **Occasional regime shifts.** The 2016 earthquake is an example: a sudden change in the
  pattern that no history predicts.

No single model family is best on all of these at once. That is why the platform builds a
deliberate **spectrum** of approaches and lets the backtest pick between them.

### 1.2 The organising idea: a spectrum of assumptions

The five tiers differ along three axes that matter more than the algorithm names.

**1. Local vs global.** A *local* model fits separate parameters to each series (1782 separate
fits). A *global* model fits one set of parameters across all series, so patterns learned from
one series help forecast the others. Montero-Manso & Hyndman (2021) show that global models are
no less expressive than local ones even when the series look very different from each other,
and in practice they often generalise better, because they have far more data per parameter.

**2. Where the model's structure comes from:**
- *assumed*: explicit equations for trend and seasonality (ETS, ARIMA);
- *engineered*: hand-built lag, calendar and promotion features (LightGBM);
- *learned*: a neural network learns its own internal representation (NHITS);
- *pre-trained*: the knowledge comes from a large corpus of *other* time series (Chronos).

**3. Training cost vs data need.** Moving along that list, models need more data or compute, and
make fewer assumptions baked in.

| Tier | Model | Local / global | Trained on our data? | Uses covariates? | Interval method |
|---|---|---|---|---|---|
| 0 | Naive, seasonal naive | local | no (a rule, not a fit) | no | crude constant band |
| 1 | AutoARIMA, AutoETS | local | yes, 1 model per series | no | conformal |
| 2 | LightGBM via `mlforecast` | **global** | yes, 1 model for all | **yes** | conformal |
| 3 | NHITS via `neuralforecast` | **global** | yes, 1 model for all | no (univariate) | conformal |
| 4 | Chronos-Bolt-Base | **global (pre-trained)** | **no, zero-shot** | no | native quantiles |

A few things apply to every tier:

- **One output format.** Every tier returns the same columns (`unique_id, ds, model_name,
  yhat, yhat_lo80, yhat_hi80`) and is scored on the same five backtest windows. The comparison
  only reflects the models, not differences in how each was tested.
- **All-zero series** (53 of them) are forecast as zero outside the model tiers. Feeding them
  to the models would only add noise and runtime.
- **Leading zeros are trimmed before fitting:** the days before a series first recorded any
  sales (not-yet-recorded, not real demand; this also covers stores' pre-opening periods). Days a
  store was closed *within* a series' life (mostly 1 January) stay in, because these models need a
  gap-free daily calendar; they are a small share of rows. The reasons are in `eda-findings.md` §2.

---

### 1.3 Tier 0 — Naive baselines

**What they are.** For a series $y_1,\dots,y_T$:

$$\text{Naive:}\quad \hat y_{T+h\mid T} = y_T$$

$$\text{Seasonal naive:}\quad \hat y_{T+h\mid T} = y_{T+h-m(k+1)},\qquad k=\left\lfloor \tfrac{h-1}{m}\right\rfloor,\ m=7$$

In words: naive repeats the last observed value; seasonal naive repeats the last observed week.

**Theory.** Naive is the optimal forecast for a *random walk*, a series where tomorrow is today
plus unpredictable noise. Seasonal naive is the seasonal version of the same idea. Neither has
parameters to estimate, so neither can overfit.

**Why they're here.** They are the *floor* every other model has to beat, not candidates for
the job. A sophisticated model that can't beat "same as last week" has not learned anything
useful. The same idea underlies MASE (mean absolute scaled error), a standard metric that
scales every model's error by the naive forecast's error. Weekly seasonality is so strong in
this data (Sunday vs Thursday is roughly 1.6×) that seasonal naive is expected to be a
genuinely hard floor.

**A simplification we're making.** Their intervals use a *constant* width, derived from the
spread of in-sample errors (PLAN.md P0.6a). The correct interval for a random walk widens with
the horizon, roughly in proportion to $\sqrt{h}$, so these bands will be too narrow at longer
horizons. That is acceptable for a baseline, and the interval-coverage metric (P0.7) will show
how far off they are.

---

### 1.4 Tier 1 — Classical statistical models (AutoETS, AutoARIMA)

Both are **local** models: fitted separately to each series, using only that series' own
history. The `statsforecast` library implements them efficiently enough to fit all 1782 series
in parallel.

#### Exponential smoothing (ETS)

ETS models a series as a combination of unobserved components: **E**rror, **T**rend and
**S**easonality. Each component can be none, additive or multiplicative, and the trend can also
be damped (it flattens out over time). Each component is updated by a smoothing recursion.
For example, with additive error, no trend and additive seasonality of period *m*:

$$
\begin{aligned}
\ell_t &= \alpha\,(y_t - s_{t-m}) + (1-\alpha)\,\ell_{t-1}\\
s_t &= \gamma\,(y_t - \ell_{t-1}) + (1-\gamma)\,s_{t-m}\\
\hat y_{t+h\mid t} &= \ell_t + s_{t+h-m(k+1)}
\end{aligned}
$$

- $\ell_t$ is the current *level* (the underlying sales rate).
- $s_t$ is the *seasonal* effect for that day of the week.
- $\alpha$ and $\gamma$ are smoothing weights between 0 and 1. Higher values mean the model
  reacts faster to recent data.

Hyndman et al. (2002) recast ETS as proper *state-space models*. That gives ETS a likelihood, so
different variants can be compared formally with AICc (a model-selection score that rewards
good fit but penalises extra parameters). **AutoETS** fits the candidate variants for each
series and keeps the one with the lowest AICc. Multiplicative forms need strictly positive
data, so on series containing zeros the search is effectively limited to additive forms.

#### ARIMA

ARIMA models the series' dependence on its own past and on past forecast errors. In backshift
notation, where $B$ shifts the series back one step ($By_t = y_{t-1}$), a seasonal
ARIMA$(p,d,q)(P,D,Q)_m$ model is:

$$\phi(B)\,\Phi(B^m)\,(1-B)^d\,(1-B^m)^D\,y_t = \theta(B)\,\Theta(B^m)\,\varepsilon_t$$

- $(1-B)^d$ and $(1-B^m)^D$ are ordinary and seasonal **differencing**: they remove trends and
  repeating seasonal patterns so the rest of the model can work on a stable series.
- $\phi, \Phi$ are **autoregressive** (AR) terms: dependence on the series' own past values.
- $\theta, \Theta$ are **moving-average** (MA) terms: dependence on past forecast errors.
- $\varepsilon_t$ is white noise, the unpredictable part.

This is the Box–Jenkins (1970) framework. **AutoARIMA** follows the Hyndman–Khandakar (2008)
algorithm:
1. Choose the differencing orders $d$ and $D$ with statistical tests (unit-root and seasonality
   tests).
2. Run a stepwise search over the remaining orders $(p,q,P,Q)$, keeping the model with the
   lowest AICc.

#### How it's configured here

- **Weekly seasonality:** both models use `season_length=7`.
- **Different history lengths:** AutoETS is fitted on each series' full history. AutoARIMA uses
  only the **last 180 days**: its stepwise search took about 2 hours per backtest fold on full
  history, versus about 7 minutes on 180 days. On a 60-series test sample, the shorter history
  cost only 0.3 WAPE points. For a 28-day horizon, the most recent six months carry most of the
  relevant dynamics, so this is a deliberate, documented scoping choice.

#### Why this tier is included

- **Strong benchmarks.** In the M-series forecasting competitions, these methods were hard to
  beat for decades. They are the "does the complexity pay off?" reference for everything above.
- **Interpretable.** Every parameter has a meaning (a smoothing weight, an autoregressive
  coefficient) that can be explained to a stakeholder.
- **Good at what dominates the volume.** The high-volume, smooth series that account for most
  sales are exactly where these methods do well.

#### Expected limitations

- **Each series learns alone.** A series cannot borrow information from similar series, which
  hurts short or noisy histories.
- **No covariates in this setup.** They are run without holiday or promotion inputs. ARIMA can
  accept external regressors, but the plan keeps this tier univariate to stay a clean
  reference.
- **Intermittent demand.** Both assume a continuous-valued series. They do poorly on
  intermittent demand, for which specialised methods exist (e.g. Croston, 1972). Those are out
  of scope here and named as a limitation (see `eda-findings.md` §2).
- **Regime shifts.** They adapt to a sudden shift like the earthquake only as fast as their
  smoothing weights allow.

---

### 1.5 Tier 2 — Global gradient-boosted trees (LightGBM via `mlforecast`)

**Forecasting reframed as regression.** Instead of modelling the time dynamics directly, this
tier turns forecasting into ordinary supervised learning. Each row (series *i*, day *t*)
becomes a training example: the target is $y_{i,t}$, and the inputs are features known before
*t*:
- lags (the value 7, 14 and 28 days earlier) and rolling means/standard deviations;
- calendar fields (day of week, month);
- holiday flags, with the observed-date logic from `eda-findings.md` §3;
- the number of items on promotion;
- the oil price;
- static store and family attributes.

These are the leakage-free features built in P0.4. **One model is fitted to all 1782 series at
once.**

**Gradient boosting** (Friedman, 2001) builds the prediction as a sum of many small decision
trees:

$$F_M(\mathbf x) = \sum_{m=1}^{M} \nu\, f_m(\mathbf x)$$

- Each new tree $f_m$ is fitted to the *pseudo-residuals* $r_i = -\,\partial L\big(y_i, F(\mathbf x_i)\big) / \partial F$:
  the direction in which the current prediction should move to reduce the loss $L$.
- $\nu$ is the learning rate: how big a step each tree takes.
- In effect this is gradient descent, carried out in the space of functions rather than
  parameters.

**LightGBM** (Ke et al., 2017) makes this fast at scale. It buckets feature values into
histograms when searching for splits, and it grows each tree leaf by leaf, always splitting
where the loss drops most. That is what makes a global model over about 3M rows practical on a
laptop.

**Multi-step forecasting.** A lag-7 feature is only known for the first 7 days of a 28-day
horizon. `mlforecast` handles this **recursively**: it predicts day $T+1$, feeds that
prediction back in as the lag for later days, and repeats. The alternative is the **direct**
strategy: one model per horizon step, using only lags that are always known (Ben Taieb et al.,
2012). The trade-off:
- *Recursive* is one model, but errors compound across the horizon.
- *Direct* avoids compounding errors, but means training 28 models.

**Why this tier is included**

- **A proven track record on the most similar problem.** The M5 competition (Makridakis et al.,
  2022) forecast Walmart retail sales, a very close analogue of this data (daily, hierarchical,
  intermittent, promotion-driven). Most top solutions were global LightGBM models.
- **Covariates come naturally.** This is the only tier here that uses holidays and promotions
  directly, and the EDA found both have measurable effects.
- **Explainable.** Feature importances show what the model relies on (reported in P0.6c).

**Expected limitations**

- **Features must be hand-built,** and feature quality caps model quality.
- **Big series dominate the loss.** Stores differ 10× in volume, so with raw targets and squared
  error the largest series drive the fit. Two open choices address this:
  - a `log1p` target transform (Open Decision #4, decided empirically in P0.7);
  - scaling each series separately (a known option, not yet specified).
- **Oil price isn't known in advance.** It is kept as a feature, but over the forecast horizon it
  is **held at its last observed value**. Feeding in the realised future price during backtesting
  would leak information a real forecaster wouldn't have. Promotions and holidays, by contrast,
  are planned ahead, so their actual future values are legitimately used.
- **Tweedie loss is not planned.** A Tweedie loss, which suits zero-inflated sales and was used
  by many M5 solutions, is a natural extension but not part of the current plan.

---

### 1.6 Tier 3 — Global deep learning (NHITS via `neuralforecast`)

**Background: N-BEATS.** Oreshkin et al. (2020) showed that a deep stack of plain fully
connected networks (MLPs) could match or beat statistical ensembles on the M4 competition. The
idea is *doubly residual stacking*:
- each block reads the input window (the last *L* days);
- it produces a **backcast** (its explanation of the input) and a **forecast** (its prediction
  for the next *h* days);
- the next block only sees the part of the input the earlier blocks failed to explain;
- the final forecast is the sum of every block's forecast:

$$\mathbf y^{(\ell+1)} = \mathbf y^{(\ell)} - \hat{\mathbf y}^{\,b}_{\ell},\qquad \hat{\mathbf y}_{T+1:T+h} = \sum_{\ell} \hat{\mathbf y}^{\,f}_{\ell}$$

**NHITS** (Challu et al., 2023) adds two ideas on top:

1. **Multi-rate input pooling.** Before each block, the input window is max-pooled (downsampled)
   with a different kernel size, so each stack sees the history at a different resolution.
   Coarse stacks see only the slow patterns; fine stacks see the day-to-day detail.
2. **Hierarchical interpolation.** Each block predicts a small number of coefficients (fewer
   than *h*), which are interpolated up to the full 28-day horizon. So low-resolution stacks
   produce smooth, trend-like shapes, high-resolution stacks add high-frequency detail, and the
   final forecast combines both.

The authors reported accuracy competitive with Transformer models on long-horizon benchmarks,
with much lower compute. Here it is configured with $h=28$ and an input window of $L=56$ days
(two horizons), and trained **globally**: training windows are sampled from across all series.

**Why this tier is included**

- **Learned rather than engineered.** The network learns cross-series temporal structure from
  the raw history instead of relying on our feature choices, so it tests whether the hand-built
  features of Tier 2 were leaving anything on the table.
- **Modern-architecture signal at low cost.** NHITS is simpler and faster to configure than
  Transformer alternatives. PatchTST (Nie et al., 2023) is the documented fallback if NHITS
  clearly underperforms (Open Decision #1; head-to-head comparison in P1.5).

**Expected limitations**

- **Sensitive to settings.** Results depend on hyperparameters, and the defaults are used in P0.
  Tuning is P1.4.
- **Less interpretable** than Tiers 1–2.
- **Runs on the full series set.** Training on all series was feasible: about 1 minute per fold
  on a laptop CPU, including conformal calibration. (The CPU was faster than the Apple GPU for
  this small network.) So no subset fallback was needed.
- **Used without covariates.** NHITS can accept future-known, past-only and static inputs. Here
  it is kept **univariate** on purpose, learning from the sales history alone. That makes it a
  clean test of learned structure against Tier 2's engineered features.
- **Needs enough history per series.** It needs an input window plus calibration windows, so the
  newest series (store 52, opened April 2017) fall back to seasonal naive in recent folds.

---

### 1.7 Tier 4 — Zero-shot foundation model (Chronos-Bolt-Base)

**The idea.** Train one large model on a big and varied corpus of *other* time series, then
forecast new series **with no training on them at all** ("zero-shot"). It is the time-series
counterpart of pre-trained language models.

**Chronos** (Ansari et al., 2024) borrows the language-model machinery directly:
1. Each series is divided by its mean absolute value, so series of any scale look alike.
2. The scaled values are quantised into a fixed vocabulary of bins, turning the series into a
   sequence of tokens.
3. A T5 language model is trained with ordinary next-token cross-entropy loss.
4. Forecasts are made by sampling future tokens.

The training corpus combines public datasets with synthetic series (Gaussian-process-generated
series and random mixtures of real ones) to improve generalisation.

**Chronos-Bolt**, the variant used here, keeps the pre-training idea but changes the mechanics:
- The input history is split into **patches** (short chunks) for a T5 encoder.
- The decoder **directly outputs several quantiles for many future steps at once**, trained with
  quantile loss, instead of sampling tokens one step at a time.
- As a result it is much faster and more accurate than the original Chronos, and it runs
  comfortably on a laptop CPU. The plan started with the *Small* model (~48M parameters). It was
  so fast (under a second for 60 series) that the project switched to the larger *Base* model
  (~205M parameters), which remains practical on CPU.
- Its quantile outputs map straight onto our output format: median → `yhat`, 10th/90th
  percentiles → the 80% interval.

**Why this tier is included**

- **A cheap strong reference.** How good is a forecast that required *zero* training on this
  retailer's data? If a zero-shot model comes close to a tuned global model, that is an
  important practical finding: cold-start series, fast prototyping, less work to maintain.
- **Current practice.** Foundation models for time series are the major 2024–26 development in
  the field, and a portfolio project should show an informed, measured view of them rather than
  either ignoring or overselling them.

**Expected limitations and scoping**

- **Univariate here.** Bolt forecasts from the series' own history only, with no promotion or
  holiday inputs. That is a real handicap on this data, and exactly the kind of trade-off the
  comparison is meant to expose. (Later Chronos releases add covariate support; this project
  uses Bolt, as the plan specifies.)
- **Runs on all series.** The plan originally restricted this tier to a stratified sample of 60
  series (the top, middle and bottom 20 by volume) to save laptop compute. In practice inference
  took under a second for 60 series, so it runs on all 1782 series like every other tier and
  competes for champion on equal terms. All-zero series are forecast as zero, as elsewhere.
- **Quality depends on pre-training.** A zero-shot model's accuracy depends on how well its
  pre-training corpus resembles our data, which we can't inspect or control.

---

### 1.8 Prediction intervals: conformal prediction

A point forecast alone can't support a stocking decision. The planner also needs to know how
wrong it might be. Tiers 1–3 produce their 80% intervals with **conformal prediction** (Vovk et
al., 2005), adapted to forecasting:

1. Run the model on several historical calibration windows (rolling-origin, like the backtest)
   and record its absolute error $|y - \hat y|$ **separately for each horizon step** $h$.
2. For horizon step $h$, take the 80th percentile $q_h$ of those errors.
3. The interval is $\hat y_{T+h} \pm q_h$.

**Why conformal.** It is *distribution-free*: it assumes nothing about the shape of the errors
(for example, that they are Gaussian, which zero-inflated sales clearly are not). It works with
any point-forecasting model, and because $q_h$ is computed per step, intervals naturally widen
at longer horizons when the model's errors do.

**The honest caveat.** Conformal prediction's coverage guarantee assumes the calibration errors
and future errors are *exchangeable*, i.e. interchangeable, with no systematic difference
between them. Time series with trends and regime shifts violate that. So the 80% here is a
target, not a guarantee, and **P0.7 measures actual coverage** against it for every tier.

- Chronos-Bolt uses its own learned quantiles instead.
- The baselines use the crude constant band described in §1.3.
- Mixing interval methods like this is itself something the coverage comparison evaluates.

---

### 1.9 Expectations to test

These are stated **before** seeing any backtest results, so the evaluation can confirm or refute
them. The model-selection rule itself is fixed and doesn't depend on them: **lowest mean WAPE
across the 5 folds wins**, with the naive baselines excluded as candidates (PLAN.md P0.8).

1. **Global beats local on aggregate accuracy.** LightGBM is expected to have the lowest overall
   WAPE (WAPE = total absolute error as a share of total actual sales), because of the M5
   evidence and because it is the only tier using promotions and holidays.
2. **Classical models are close on the big, smooth series.** The gap between tiers is expected to
   be smallest for the high-volume families (e.g. GROCERY I, BEVERAGES) and largest for
   intermittent ones.
3. **Zero-shot is surprisingly competitive, but no winner.** Chronos-Bolt is expected to beat the
   seasonal-naive floor clearly on its 60-series sample, without matching the covariate-aware
   global models. *(Kept as originally written. Since then the tier's scope changed to all series
   and Chronos-Bolt-Base; the evaluation tests the expectation on that scope.)*
4. **Every tier fails on intermittent series.** For families like BOOKS or BABY CARE (over 90%
   zeros), point-forecast error metrics are expected to be near-meaningless for all tiers. That
   points to a different framing (will it sell at all?) rather than a better regressor.
5. **Coverage falls short of 80% during regime shifts,** for every interval method.

Results — confirming or overturning these — will be reported in `model-evaluation.md`.

---

## References

- Ansari, A. F. et al. (2024). *Chronos: Learning the Language of Time Series.* Transactions on
  Machine Learning Research.
- Ben Taieb, S., Bontempi, G., Atiya, A. F., & Sorjamaa, A. (2012). A review and comparison of
  strategies for multi-step ahead time series forecasting based on the NN5 forecasting
  competition. *Expert Systems with Applications*, 39(8).
- Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control.*
  Holden-Day.
- Challu, C. et al. (2023). NHITS: Neural Hierarchical Interpolation for Time Series Forecasting.
  *AAAI Conference on Artificial Intelligence.*
- Croston, J. D. (1972). Forecasting and stock control for intermittent demands. *Operational
  Research Quarterly*, 23(3).
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of
  Statistics*, 29(5).
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.).
  OTexts. <https://otexts.com/fpp3/>
- Hyndman, R. J., & Khandakar, Y. (2008). Automatic time series forecasting: the forecast package
  for R. *Journal of Statistical Software*, 27(3).
- Hyndman, R. J., Koehler, A. B., Snyder, R. D., & Grose, S. (2002). A state space framework for
  automatic forecasting using exponential smoothing methods. *International Journal of
  Forecasting*, 18(3).
- Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS.*
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results,
  findings, and conclusions. *International Journal of Forecasting*, 38(4).
- Montero-Manso, P., & Hyndman, R. J. (2021). Principles and algorithms for forecasting groups of
  time series: Locality and globality. *International Journal of Forecasting*, 37(4).
- Nie, Y. et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.
  *ICLR.*
- Oreshkin, B. N. et al. (2020). N-BEATS: Neural basis expansion analysis for interpretable time
  series forecasting. *ICLR.*
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.*
  Springer.
