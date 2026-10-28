# Model Card: Gaussian Process-Based Black-Box Optimisation

**Approach:** Gaussian Process (GP) surrogate models with acquisition-based query selection  
**Type:** Sequential, derivative-free numerical optimisation  
**Version:** 1.1, 28 October 2026  
**Status:** Evaluations completed through Week 9; Week 10 queries proposed, with results pending

This card focuses on my modelling approach and its performance. The [README](README.md) introduces the project and records the strategy used in Weeks 1–3. The [dataset datasheet](data_sheet.md) covers the observations, their collection and preprocessing, and the restrictions on sharing them.

## Intended use

I developed the approach to maximise eight unknown numerical functions with a limited number of evaluations. GP-based optimisation is useful when an objective can be queried but its formula or gradients are unavailable, and when new evaluations are costly enough to justify fitting a surrogate.

The fitted models are specific to this capstone. They should not be used to predict unrelated data, treated as proof of a global optimum, or transferred directly to high-stakes decisions. Highly discontinuous, noisy or higher-dimensional problems may require different models and additional safeguards.

## Model and decision process

I fitted a separate `GaussianProcessRegressor` (scikit-learn) to each function rather than forcing the same settings across all eight. The model takes previously evaluated input vectors and objective values, then estimates a **predicted mean** and **standard deviation** at new inputs. The input ranges and dataset structure are described in the [README](README.md) and [datasheet](data_sheet.md).

Most of my queries were guided by the Upper Confidence Bound (UCB) acquisition function:

$$
\operatorname{UCB}(x) = \mu(x) + \beta\sigma(x)
$$

Here, $\mu(x)$ is the predicted mean, $\sigma(x)$ represents uncertainty, and $\beta$ controls the emphasis on exploration. I generally selected a high-scoring candidate, while also checking its distance from earlier observations, whether it sat in a promising region, and how sensitive the recommendation was to the kernel. The highest acquisition score was not always the final choice.

### How the strategy developed

My initial GP and UCB approach, and the changes made during **Weeks 1–3**, are described in the [README](README.md). During **Weeks 4–9**, I put more weight on checking whether the surrogates predicted *new* observations well. I compared RBF and Matérn kernels, isotropic and separate length scales, and different noise assumptions. Leave-one-out cross-validation (LOOCV) was useful for comparing models, but chronological backtesting was particularly relevant because it tested predictions using only earlier information.

As the functions began to behave differently, I moved away from using a common search strategy. I used local refinement where previous queries had identified a credible peak, and wider exploration where nearby queries had stopped improving the result. I also introduced minimum-distance constraints and local trust regions. Function 5 benefited from modelling square-root-transformed outputs; for Functions 7 and 8, I used ensembles to take account of differences between GP models.

For **Week 10**, the following were my selected models and intended query strategies. These describe the next submissions, **not evaluated Week 10 outcomes**.

| Function | Proposed Week 10 model | Query selection |
|---|---|---|
| 1 | Isotropic Matérn 3/2 | UCB, $\beta=1.96$; broader coverage |
| 2 | Anisotropic RBF + `WhiteKernel` | UCB, $\beta=1.5$; explore a less-tested region |
| 3 | Isotropic Matérn 3/2 | UCB, $\beta=0.75$; local refinement |
| 4 | Matérn 3/2 + amplitude + `WhiteKernel` | UCB, $\beta=1.5$; constrained search |
| 5 | Matérn 5/2 + amplitude, fitted to $\sqrt{y}$ | UCB, $\beta=0.5$, alongside a controlled boundary test |
| 6 | Matérn 3/2 + amplitude + `WhiteKernel` | UCB, $\beta=1.0$; refine the improved region |
| 7 | Equal-weight RBF / Matérn 3/2 ensemble | Ensemble UCB, $\beta=0.5$; minimum separation |
| 8 | Anisotropic RBF / anisotropic Matérn 5/2 / isotropic RBF ensemble (20% / 50% / 30%) | Ensemble UCB, $\beta=0.5$; local trust region |

The specific choices reflect different evidence: Functions 6–8 had improved their best outputs in Week 9, whereas Function 2 had not improved on its Week 4 maximum. For Function 5, I proposed the boundary test at `(0.95, 1, 1, 1)` to investigate repeated near-identical outputs, rather than automatically submitting the highest-scoring UCB candidate.

## Performance

My main **optimisation metric** was the best observed objective value, compared with the best value in the initial data. The table shows the results after nine weekly submissions per function; the number of evaluations and their origins are documented in the [datasheet](data_sheet.md).

| Function | Best initial output | Best observed through Week 9 | First achieved |
|---|---:|---:|---|
| 1 | 7.71 × 10⁻¹⁶ | 7.71 × 10⁻¹⁶ | Initial data |
| 2 | 0.611205 | **0.623849** | Week 4 |
| 3 | −0.034835 | **−0.000133** | Week 4 |
| 4 | −4.025542 | **0.625128** | Week 5 |
| 5 | 1,088.859618 | **8,662.482500** | Week 7 |
| 6 | −0.714265 | **−0.086816** | Week 9 |
| 7 | 1.364968 | **2.425432** | Week 9 |
| 8 | 9.598482 | **9.982223** | Week 9 |

Seven functions improved on their initial best values. Function 1 remained difficult because nearly all outputs were effectively zero. The functions have very different output scales, so their raw objective values should not be averaged or directly compared. Their true maxima are unknown.

For **surrogate accuracy**, I considered LOOCV and chronological-backtesting RMSE, individual prediction errors, and the uncertainty attached to each prediction. Week 9 provides one direct comparison of forecasts against subsequently observed outputs:

| Function | Week 9 forecast | Actual output | Absolute error |
|---|---:|---:|---:|
| 1 | −0.000344 | ≈ 0 | 0.000344 |
| 2 | 0.557177 | 0.551466 | 0.005711 |
| 3 | −0.036778 | −0.032717 | 0.004061 |
| 4 | 0.263490 | 0.030650 | 0.232839 |
| 5 | 7,926.10 | 7,786.371350 | 139.73 |
| 6 | −0.255509 | −0.086816 | 0.168693 |
| 7 | 2.215546 | 2.425432 | 0.209886 |
| 8 | 9.965622 | 9.982223 | 0.016601 |

These forecasts were recorded before the Week 9 outputs were received, rather than fitted retrospectively. They show that predictive accuracy and optimisation success are different: Functions 6 and 7 produced new maxima despite imperfect predictions. I have not presented a combined RMSE across all eight functions because their output scales differ, nor claimed a measured uncertainty-coverage rate without a consistent calculation.

## Assumptions, limitations and trade-offs

The approach assumes that the functions are regular enough for nearby observations to be informative. RBF and Matérn kernels encode different smoothness assumptions, so an unexpected narrow peak or abrupt change may be missed. Limited observations, particularly in the higher-dimensional functions, also leave substantial parts of the search space unexplored. Because later inputs were selected adaptively, the data are concentrated around promising regions, which can make validation results look stronger than performance elsewhere.

More flexible kernels and ensembles sometimes improved retrospective fit, but they also introduced more parameters and opportunities to overfit. LOOCV results did not always agree with chronological backtesting. I therefore treated the observed outputs as the evidence of improvement and the surrogate predictions as guidance, not guarantees. The recurring practical trade-off was whether to spend a query refining a known peak or testing an uncertain region.

## Transparency, ethics and reproducibility

The functions involve numerical observations rather than personal records, so privacy and group fairness were not the main risks in this exercise. The more relevant issue is **transparency about uncertainty and decisions**. I distinguish observed outputs from predictions and proposed queries, and aim to record kernel settings, transformations, acquisition parameters, random seeds, search constraints and any manual changes to an acquisition-based recommendation. Those details make it possible to reproduce a decision and examine whether its assumptions were reasonable.

In a real application, I would need to validate the surrogate under the relevant operating conditions and consider the consequences of poor recommendations, rather than relying on a high UCB score alone. The [datasheet](data_sheet.md) covers dataset provenance and permissions for any public release of the observations.
