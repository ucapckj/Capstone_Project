# Black-Box Optimisation Capstone Project

**Professional Certificate in Machine Learning and Artificial Intelligence**  
**Imperial College Business School, London**  
**Project status:** All 13 weekly evaluation rounds completed (November 2026)  
**Related documents:** [Dataset datasheet](data_sheet.md) | [Model card](model_card.md)

## Project overview

In this project, I tried to find the best inputs for eight mathematical functions without initially knowing how the functions worked. Each week I could test only one new input per function, so I used machine learning to estimate which experiments might be most useful. I started with Gaussian Process models, then adjusted my methods as the results came back. Several functions improved substantially. For one function, I discovered that the best result occurred at the upper boundary; for two others, I identified their mathematical structure and reached their known or calculated maxima. The project showed me why checking assumptions matters as much as refining models.

## 1. Problem and approach

The capstone follows the *approach* of the NeurIPS 2020 Black-Box Optimization for Machine Learning competition; I do not assume that all eight challenge functions were taken from that competition. There were eight separate functions, with 2 to 8 inputs, each restricted to the interval [0, 1]. The objective was to **maximise each function** under a limited budget: 175 programme-provided starting observations followed by one evaluated query for each function in each of 13 rounds.

I initially used scikit-learn's `GaussianProcessRegressor` to estimate an output and uncertainty at untested inputs. Most early queries were guided by an Upper Confidence Bound (UCB) acquisition function:

$$
\mathrm{UCB}(\mathbf{x})=\mu(\mathbf{x})+\beta\sigma(\mathbf{x})
$$

Here, $\mu$ is the GP's predicted mean, $\sigma$ is its predictive standard deviation and $\beta$ controls the emphasis on uncertainty. I progressively compared RBF and Matérn kernels, isotropic and anisotropic length scales, noise assumptions, output transformations and ensembles. I also used hybrid global/local sampling, differential evolution, minimum-distance constraints and local refinement. The [model card](model_card.md) explains the model choices, performance and limitations without repeating the full weekly narrative.

**Submission precision:** New input coordinates were submitted to **six decimal places**. The initial observations retain the precision of the supplied datasets.

## 2. Final results

The table uses **my best observed value across the initial data and all 13 weekly evaluations**, rather than the official leaderboard, which did not incorporate some late submissions. Higher is better for every function. “Best week” refers to the week in which that maximum was first observed.

| Function | Dimensions | Best from initial dataset | Final best observed | Best week |
|:--|--:|--:|--:|:--|
| 1 | 2 | 7.71 × 10⁻¹⁶ | 0.000006756159 | 13 |
| 2 | 2 | 0.611205 | 0.670534 | 12 |
| 3 | 3 | −0.034835 | −0.000123351 | 13 |
| 4 | 4 | −4.025542 | 0.625128 | 5 |
| 5 | 4 | 1,088.860 | 8,662.482500 | 7 |
| 6 | 5 | −0.714265 | −0.001592738 | 10 |
| 7 | 6 | 1.364968 | 3.322368011 | 13 |
| 8 | 8 | 9.598482 | 10.000000 | 11 |

The values are not directly comparable across functions because their scales differ. For Function 5, `(1, 1, 1, 1)` gave the highest observed output; later tests supported a permutation-symmetric response. Function 7 was identified as the standard **Hartmann-6** benchmark, and its Week 13 input reached the known maximum. For Function 8, a recovered separable quadratic predicted the maximum of **10**, confirmed in Week 11 and tested again in Weeks 12–13. These structural identifications are different from the earlier GP-guided optimisation results.

## 3. How the work developed

The notebooks retain the experimental code and the decisions leading to each weekly submission. The summary below highlights changes rather than reproducing every query.

| Round | Main development and lesson |
|:--|:--|
| **1** | Established a GP–UCB baseline and made the first additional submissions. |
| **2** | Tuned kernel length scales and increased exploration in several functions; candidate searches moved beyond small grids. |
| **3** | Introduced hybrid global/local sampling and compared isotropic with anisotropic RBF kernels. |
| **4** | Reduced exploration on selected functions. Function 3 achieved a strong local result (approximately −0.000133). |
| **5** | Matérn models helped identify Function 4's personal best (0.625128); Functions 5, 7 and 8 also improved. |
| **6** | Function 5 climbed above 8,135, while Function 6 improved to −0.150876. Prediction checks and function-specific settings became more important. |
| **7** | Function 5 reached **8,662.4825** at the upper corner. Differential evolution, distance constraints and ensembles were increasingly used elsewhere. |
| **8** | No function improved its running best. The Function 5 test `(1, 0.95, 1, 1)` began a series investigating coordinate symmetry. |
| **9** | Functions 6–8 improved. Function 7 moved to a Matérn GP and reached 2.425432; the second Function 5 symmetry test matched Week 8's output. |
| **10** | Function 6 achieved **−0.001593** with a noise-aware, constrained GP–UCB search. A broad Function 2 exploration performed poorly. |
| **11** | Function 8's quadratic model predicted the input that returned **10.0**. Function 5's four single-coordinate symmetry tests were complete. |
| **12** | Function 2 reached its personal best (0.670534). Function 7's six-GP ensemble found a new best (2.477715). Function 8's quadratic passed a deliberate perturbation test. |
| **13** | Function 7's identified Hartmann-6 formula led to **3.322368011**. Functions 1 and 3 reached new personal bests; Functions 5 and 8 were tested further for structural accuracy. |

The main change in my approach was moving from a largely generic GP acquisition procedure to **function-specific decisions informed by model errors, observed patterns and deliberate tests**. The limitations were equally informative. For example, Function 2's final ensemble was overly optimistic, and tiny local adjustments around Function 6's best point could return substantially worse results.

## 4. Repository contents

```text
.
├── README.md                    Project overview, approach and results
├── data_sheet.md                Data provenance, composition and limitations
├── model_card.md                Modelling strategies, performance and limitations
├── requirements.txt             Python environment
├── function_1/ ... function_8/
│   ├── function_N.ipynb         Weekly experiments for each function
│   ├── initial_inputs.npy       Starting input observations
│   └── initial_outputs.npy      Starting objective values
├── submissions/
│   ├── inputs.txt               Weekly submitted inputs
│   ├── outputs.txt              Weekly returned outputs
│   ├── weekly_submissions.csv   One row per evaluated query (104 rows)
│   └── best_weekly_results.csv  Best result achieved by function
└── scripts/
    ├── check_data.py            Optional data consistency checks
    └── check_weekly_data.py     Optional weekly-record checks
```

The eight notebooks are organised chronologically, with a common imports/setup cell, weekly model experiments and the evaluated queries. Alternative configurations are included when relevant to the choice of submission.

## 5. Running the notebooks

1. Install the dependencies with `python -m pip install -r requirements.txt`.
2. Open Jupyter from the repository folder, or open a notebook directly from its `function_N/` directory.
3. Run the setup cell, then work through the weekly sections in order. The notebooks reuse fitted models and data arrays across rounds, and some earlier candidate searches require substantial memory. Review these cells before using **Run All**.
4. The full 13-week input and output history is in `submissions/weekly_submissions.csv`. The Week 13 outputs are shown at the end of the respective notebooks.

The optional data consistency checks can be run with `python scripts/check_data.py` or `python scripts/check_weekly_data.py`.

## 6. Data availability and reuse

The initial observations and black-box evaluation platform were supplied through the programme. The initial `.npy` arrays and the weekly query/output history are included in the complete repository package. Public distribution is subject to the programme's data-sharing terms. If public release is not permitted, describe the restricted datasets and provide authorised access details instead of hosting them. Large datasets should be kept outside GitHub; this project's numerical observations are small.

The notebooks document exploratory educational work, not a production optimisation service. See [Data access and reuse](DATA_ACCESS.md) for publication considerations. The [datasheet](data_sheet.md) records the sampling and distribution caveats, and the [model card](model_card.md) explains assumptions, validation and responsible reuse.
