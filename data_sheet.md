# Dataset Datasheet: Black-Box Optimisation Capstone

**Professional Certificate in Machine Learning and Artificial Intelligence**  
**Imperial College Business School, London**  
**Dataset status:** 13 weekly evaluation rounds completed; latest recorded results through Week 13 (November 2026)

## Motivation and provenance

I collected and used this dataset as part of the programme's black-box optimisation capstone. The project follows the approach used in the NeurIPS 2020 *Black-Box Optimization for Machine Learning* competition, although this does not establish that all eight functions were original NeurIPS competition benchmarks.

The task was to find high-output input vectors for eight numerical functions with initially unknown equations. The programme provided starting observations and access to an evaluation platform. I chose one further input per function each week over 13 rounds and recorded the returned outputs. The dataset supports the reconstruction of those decisions, comparison of optimisation approaches and evaluation of surrogate-model predictions. It is not a general-purpose real-world dataset or a supervised classification benchmark.

## Composition

Each observation contains one input vector $\mathbf{x}$ with $d$ continuous coordinates and one scalar output $y$. All coordinates are bounded between 0 and 1. The eight functions are independent optimisation tasks and have markedly different output scales.

| Function | Input dimensions | Initial observations | Weekly evaluations | Total |
|:--|--:|--:|--:|--:|
| 1 | 2 | 10 | 13 | 23 |
| 2 | 2 | 10 | 13 | 23 |
| 3 | 3 | 15 | 13 | 28 |
| 4 | 4 | 30 | 13 | 43 |
| 5 | 4 | 20 | 13 | 33 |
| 6 | 5 | 20 | 13 | 33 |
| 7 | 6 | 30 | 13 | 43 |
| 8 | 8 | 40 | 13 | 53 |
| **Total** | | **175** | **104** | **279** |

The numerical history contains 104 weekly inputs and their corresponding returned outputs. The starting datasets contain the 175 programme-provided observations.

The functions contain no apparent personal or organisational records. That does not remove possible programme restrictions on redistribution.

## Collection process and sampling considerations

The programme chose the initial observations; I do not know their original selection procedure. For each subsequent round, I selected a candidate using the observations then available. Earlier queries typically used GP-based acquisition and broader sampling, while later queries increasingly used local refinement, explicit structural experiments or direct optimisation of an identified equation.

Consequently, the 104 additional observations are **adaptively selected**, not a random sample of the full input spaces. Promising regions are oversampled, while some regions remain sparsely explored. A model validated on these observations may therefore appear more reliable than it would be over the whole domain.

The weekly query and result history is recorded in `submissions/weekly_submissions.csv`, with the submitted text histories retained under `submissions/`. Each CSV row identifies the round, function, input dimensions, coordinates and returned output.

## Preparation and transformations

I paired each submitted input with its corresponding evaluator output and preserved the numerical values. The programme required submitted input coordinates to be given to **six decimal places**; these are the coordinates used for the weekly history. The initial observations retain the numerical precision of the supplied arrays.

For modelling, I combined the initial and accumulated weekly observations separately for each function. I used GP target normalisation and compared alternative kernels and noise assumptions. I also tested output transformations, including square-root transformations for Function 5 and transformed-output models for selected other functions. Those transformed targets were modelling choices, not replacements for the returned outputs reported in this dataset.

For evaluation, I used observed maxima, individual prediction errors, leave-one-out cross-validation and chronological backtesting where applicable. The [model card](model_card.md) gives the modelling interpretation and final outcomes; the [README](README.md) records the weekly decisions.

## Uses and limitations

The intended use is to study sequential black-box optimisation when objective evaluations are limited, and to assess how surrogate-model assumptions and query-selection strategies affect outcomes.

The dataset is small relative to the dimensionality of several functions, especially Functions 7 and 8. There is also substantial variation in output scale, so errors and raw objective values should not be compared or averaged across all eight functions without appropriate scaling. Some apparently nearby observations returned very different outputs; without repeated identical-input tests, these differences alone cannot establish observation noise.

Later experiments identified additional structure: Function 5 showed strong permutation-symmetry evidence; Function 7 matched Hartmann-6; Function 8 matched a separable quadratic to numerical precision. These findings apply to the relevant functions and do not imply that the remaining functions have known formulas or that every observed best is a proven global maximum.

The data should not be treated as representative of a real-world population, used to infer causation, or assumed to justify high-stakes decisions without independent validation.

## Distribution and maintenance

Each function folder includes `initial_inputs.npy` and `initial_outputs.npy`. The further queries and outputs appear in `submissions/weekly_submissions.csv` and the accompanying text records. The full numerical dataset contains 279 evaluated input-output pairs.

This is educational challenge data. I have not established whether the programme permits public redistribution of its starting observations and returned outputs. Before publishing the dataset in a public repository, I would confirm the relevant programme terms. Where redistribution is restricted, the data should be described and linked through an authorised access route rather than hosted directly.

I maintain the project documentation and submission history. The observed outputs remain separate from surrogate predictions and reconstructed mathematical relationships. The dataset covers all 13 completed weekly rounds.
