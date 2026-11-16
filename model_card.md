# Model Card: GP-Based and Structure-Aware Black-Box Optimisation

**Approach:** Gaussian Process (GP) surrogates, acquisition-based query selection and later function-specific structural analysis  
**Type:** Sequential numerical optimisation over bounded continuous domains  
**Version:** 2.0, 15 November 2026  
**Status:** All 13 weekly evaluations completed

This model card focuses on how I selected candidate inputs, what worked and where the approach was unreliable. The [README](README.md) records the week-by-week development and repository instructions; the [dataset datasheet](data_sheet.md) documents the observations, preprocessing, distribution and limitations.

## Intended use

I developed the approach to maximise eight initially unknown numerical functions while making only one additional evaluation per function per week. Gaussian Process optimisation is suitable for bounded, expensive-to-evaluate problems where previously observed inputs and outputs can guide a new query. The experiments are specific to this educational capstone, not a deployable model for unrelated data.

The approach should not be assumed to work for highly discontinuous or very high-dimensional objectives without further checks. It should not be used for high-stakes applications without validated constraints, risk controls and uncertainty estimates. A high acquisition score is not evidence of a guaranteed improvement.

## Model and decision process

For each function, I fitted a separate scikit-learn `GaussianProcessRegressor`. A GP predicts the mean $\mu(\mathbf{x})$ and standard deviation $\sigma(\mathbf{x})$ at a candidate input. Most early rounds used Upper Confidence Bound (UCB) acquisition:

$$
\mathrm{UCB}(\mathbf{x})=\mu(\mathbf{x})+\beta\sigma(\mathbf{x})
$$

Higher $\beta$ gives more weight to exploration; lower values favour high predicted outputs. I initially used isotropic RBF kernels and subsequently compared anisotropic RBF, Matérn 3/2 and Matérn 5/2 kernels. Other changes included target transformations, explicit noise components, GP ensembles, differential evolution, hybrid candidate sampling, minimum-distance penalties and local trust regions.

My candidate decisions were not always the unmodified maximum of UCB. When the results suggested a specific hypothesis, I chose controlled tests instead. This distinction became increasingly important in the final rounds, when Function 5's symmetry and the mathematical identities of Functions 7 and 8 were investigated.

### Function-specific developments

| Function | Main approach and finding |
|:--|:--|
| **1** | Began with GP–UCB and distance-based exploration; later directional analysis and a log-magnitude fit identified the Week 13 personal best. Most earlier outputs were effectively zero. |
| **2** | Compared anisotropic RBF and Matérn models, noise assumptions and a weighted GP ensemble. The Week 12 maximum improved on the initial best, but the final ensemble overpredicted the Week 13 result. |
| **3** | Early RBF exploitation produced a strong Week 4 result; later Matérn and transformed-output comparisons supported a small Week 13 local refinement. |
| **4** | A Matérn 3/2 GP with hybrid sampling found the Week 5 maximum. Later interpolating and regularised models disagreed substantially near that solution. |
| **5** | Square-root-transformed GP modelling and testing the upper corner found **8,662.4825** in Week 7. Subsequent permutations and coordinate tests supported symmetry; a fitted parametric model closely predicted Week 13. |
| **6** | A noise-aware Matérn GP with constrained differential evolution selected the unusually strong Week 10 query. Later very close inputs returned lower outputs than expected, exposing local predictive unreliability. |
| **7** | RBF/Matérn GPs and then a six-GP ensemble found improvements through Week 12. The data were subsequently identified as standard Hartmann-6; direct optimisation produced the Week 13 optimum. |
| **8** | Anisotropic GP ensembles guided the early search. A 17-parameter separable quadratic fitted to the initial observations explained held-out weekly results to numerical precision and gave the Week 11 maximum of 10. |

## Performance

The main **optimisation metric** is the highest observed objective value, compared with the highest value among the initial observations. This table includes all 13 weekly evaluations, regardless of whether late submissions appeared in the official leaderboard.

| Function | Best from initial data | Best observed overall | Week first achieved |
|:--|--:|--:|:--|
| 1 | 7.71 × 10⁻¹⁶ | 0.000006756159 | 13 |
| 2 | 0.611205 | **0.670534** | 12 |
| 3 | −0.034835 | **−0.000123351** | 13 |
| 4 | −4.025542 | **0.625128** | 5 |
| 5 | 1,088.860 | **8,662.482500** | 7 |
| 6 | −0.714265 | **−0.001592738** | 10 |
| 7 | 1.364968 | **3.322368011** | 13 |
| 8 | 9.598482 | **10.000000** | 11 |

The output scales differ considerably, so it would be misleading to average these raw scores. **All eight functions improved on their initial best values by the completion of Week 13.**

My **surrogate assessment** considered leave-one-out cross-validation, chronological backtesting, individual prediction errors and differences between model configurations. Some representative decisions were:

- **Function 2, Week 13:** The weighted ensemble predicted approximately **0.727**, but the evaluator returned **0.630**. This was below the Week 12 personal best of **0.671**.
- **Function 4, Week 13:** Five GP models predicted markedly different values at one candidate, approximately **0.355 to 1.081**; the actual output was **0.522**.
- **Function 6, Week 10:** The GP predicted a mean near **−0.128**, but the selected candidate returned **−0.001593**. Acquisition success did not mean accurate prediction.
- **Function 7, Week 13:** The standard Hartmann-6 formula evaluated at the submitted six-decimal input matched the returned **3.322368011** to floating-point precision. This was a *direct mathematical-model* prediction, not a GP forecast.
- **Function 8, Week 11:** A quadratic fitted using the 40 initial observations gave a held-out mean absolute error around **1.67 × 10⁻¹⁴** on the first ten weekly evaluations. The calculated optimum returned exactly **10**; two subsequent perturbations also matched predictions to the reported precision.

These examples show why I distinguish **optimisation outcome** from **surrogate accuracy**. I do not claim a single aggregate prediction RMSE or uncertainty-coverage rate across the eight differently scaled functions.

## Assumptions, limitations and trade-offs

The GP models assumed enough regularity in each response surface for observations to be informative at nearby inputs. RBF and Matérn kernels embody different smoothness assumptions, and the available observations were sparse in higher dimensions. Anisotropic kernels and ensembles added flexibility, but also increased the number of choices that could overfit limited data.

Noise was another source of uncertainty. Adding a `WhiteKernel` changes the assumed noise level, but it does not prove that a function is stochastic. Large changes between nearby inputs can also reflect steep local structure or a misspecified kernel. Repeat evaluations at identical inputs would have helped distinguish these explanations.

The most important practical trade-off was **exploration versus exploitation** under a small evaluation budget. Function 5 rewarded concentrating on high-valued boundaries, whereas broad exploration in Function 2's Week 10 query did not. In Functions 7 and 8, identifying a mathematical formula eventually superseded continued GP acquisition. For Function 5, the fitted symmetric equation is an empirical approximation; unlike the identified Hartmann-6 and quadratic structures, I have not treated it as an exactly recovered hidden function.

## Transparency, responsible use and reproducibility

The dataset contains numerical function evaluations rather than personal records, so privacy and group fairness were not the primary concerns. The main responsibility is to distinguish **observations, predictions, hypotheses and identified mathematical relationships**, and to document when a recommendation was manually changed.

The notebooks record the weekly modelling decisions, parameters, candidate-generation methods and evaluated inputs. The [README](README.md) links to the 104-row query history and explains how to run the notebooks. Reproducibility depends on package versions, random seeds, candidate constraints and access to the relevant programme datasets.

For real-world use, I would validate predictions under the relevant operating conditions, investigate noise and constraints, and consider the costs of an incorrect recommendation. These methods should support, not replace, domain expertise and careful validation.
