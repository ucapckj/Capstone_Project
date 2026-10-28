# Black-Box Optimisation Capstone Project

**Professional Certificate in Machine Learning and Artificial Intelligence**  
**Imperial College Business School, London**  
**Progress:** Week 10 input submission; results available through Week 9 (28 October 2026)**
**Related document:** [data sheet](data_sheet.md) [model card](model_card.md)

## 1. Project overview

This capstone explores **black-box optimisation (BBO)**: finding inputs that maximise an unknown function when its mathematical expression and gradients are not available. I can evaluate a candidate input and observe the output, but each evaluation is limited, so the challenge is to make each new query useful.

The project contains eight separate numerical functions, with between two and eight input dimensions. I have mainly used **Gaussian Process Regression (GPR)** to estimate their behaviour and Bayesian optimisation techniques to select subsequent queries. The project draws on the approach used in the NeurIPS 2020 *Black-Box Optimization for Machine Learning* competition; further background is in the [dataset datasheet](data_sheet.md).

This README records how my approach changed over the ten rounds. The [model card](model_card.md) covers the individual surrogate models, their performance and limitations, while the [datasheet](data_sheet.md) documents the observations and their collection.

## 2. Problem and approach

Each function accepts a vector of continuous inputs, with every coordinate constrained to the interval from 0 to 1, and returns one numerical objective value:

$$
\mathbf{x}=(x_1,\ldots,x_d),\qquad y=f(\mathbf{x}),\qquad 0\leq x_i\leq 1.
$$

My objective is to **maximise** each function, not necessarily to reconstruct its entire response surface. For each weekly round, I select one input vector per function. Submitted coordinates must be specified to **six decimal places**. I then use the returned outputs to update the models before the next round.

I began with the programme-provided initial observations and fitted a separate Gaussian Process to each function. A GP is useful here because it estimates both an expected output and uncertainty at untested inputs. Much of my query selection has been based on the Upper Confidence Bound (UCB) acquisition function:

$$
\operatorname{UCB}(\mathbf{x})=\mu(\mathbf{x})+\beta\sigma(\mathbf{x}).
$$

A larger exploration parameter, \(\beta\), gives more weight to uncertainty; a smaller value favours inputs with higher predicted outputs. Over time, I also considered how close a candidate was to previous evaluations and whether different kernel models supported the same region. The highest UCB score was not always the most informative query.

## 3. Progress by week

### Week 1: Establishing a baseline

I fitted initial GP surrogate models and used UCB to guide the first additional queries. This gave me a more systematic starting point than sampling at random, although I still had limited evidence about the shape of each function.

### Week 2: Refining the assumptions

After receiving the first new outputs, I examined the effect of GP kernel length scales and the UCB exploration parameter. Five functions had improved on their initial best values by this point, but the results also showed that a reasonable-looking surrogate did not always identify a better input.

### Week 3: Starting to distinguish the functions

I began to consider whether a broad search or a more local query was appropriate for each function. The results were quite different across the eight problems, which made a single set of GP parameters increasingly difficult to justify. Functions 4 and 5 reached new best values in this round.

### Week 4: Testing sensitivity to nearby inputs

The fourth submissions improved the best observed values for Functions 2, 3, 5 and 6. Function 3 reached approximately **−0.000133**, while Function 5 improved to around **2,909.9**.

Function 4 offered a useful warning: its output moved from approximately **+0.260** in Week 3 to **−0.301** in Week 4 at relatively nearby inputs. This suggested that parts of its response surface were changing more sharply than my original GP assumptions allowed. I began giving more attention to alternative kernels and local sensitivity, rather than trusting the fitted mean on its own.

### Week 5: Comparing kernels more seriously

New maxima were obtained for Functions 4, 5, 7 and 8. Function 4's result of approximately **0.625128** was especially encouraging after the previous week's decline. Function 5 reached about **4,727.1**.

I increasingly compared RBF and Matérn kernels. The smoother RBF was not necessarily the best choice for functions with sharper local changes. I also became more cautious about changing several modelling settings simultaneously, since doing so made it harder to understand which change had helped.

### Week 6: Checking predictions against new results

Functions 5, 6, 7 and 8 improved again. Function 5 reached approximately **8,135.8**, and Function 6 achieved a less-negative maximum of approximately **−0.150876**.

I started relying more heavily on **leave-one-out cross-validation (LOOCV)** and comparisons with predictions made before new results were known. I wanted to distinguish a model that fitted the existing observations well from one that could actually guide the next evaluation. The results also supported using different exploration settings and search regions for different functions.

### Week 7: Refining promising regions

Function 5 reached **8,662.4825** at the upper corner `(1, 1, 1, 1)`. Functions 7 and 8 also set new maxima, of approximately **2.027249** and **9.964165**, respectively.

These results made local refinement increasingly attractive for some functions. At the same time, I did not want to assume that finding a strong local result meant the global optimum had been found. I continued considering broader candidates where the models were uncertain or where progress had stalled.

### Week 8: Learning from unsuccessful queries

**None of the eight functions improved its best observed output in Week 8.** This was an important counterpoint to the earlier gains. In particular, Function 7 returned a weaker-than-expected result near a previously successful region, exposing a limitation in the GP's local predictions.

Rather than simply increasing the complexity of the models, I compared alternative kernel choices, validation errors and uncertainty estimates. I also used minimum-distance constraints and local search regions where appropriate, to avoid repeatedly evaluating nearly identical inputs without a clear reason.

For Function 5, reducing the second input from 1 to 0.95, while holding the other coordinates at 1, produced **7,786.371350**. This provided a useful controlled comparison with the Week 7 maximum.

### Week 9: Updating the strategy after new evidence

Week 9 produced new maxima for three functions:

- **Function 6:** **−0.086816**, improving on Week 6.
- **Function 7:** **2.425432**, a substantial improvement on its Week 7 maximum.
- **Function 8:** **9.982223**, a smaller improvement on its Week 7 result.

Other functions demonstrated why **prediction accuracy and optimisation success are not the same**. For example, the Week 9 predictions for Functions 2 and 3 were reasonably close to the returned outputs, but neither query improved the best value. Function 4's actual output was considerably below its GP forecast.

For Function 5, changing the third input to 0.95 returned the same output as the Week 8 test of the second input. This led me to plan another controlled test of input sensitivity for the final round. For Functions 7 and 8, I also began using GP ensembles to avoid relying entirely on one kernel's assumptions.

### Week 10: Selecting the final inputs

I used all observations returned through Week 9 to make the final query choices. The strategies were deliberately different: some inputs refine a strong region, while others test a part of the search space that the earlier queries had not explored sufficiently.

The following are the **inputs chosen for the Week 10 submission**, with all coordinates shown to six decimal places:

| Function | Week 10 input vector | Reason for selection |
|---|---|---|
| 1 | `(0.000000, 0.000000)` | Explore a distant, untested corner because most observed outputs are effectively zero. |
| 2 | `(0.973000, 0.400000)` | Move away from the repeatedly sampled region around the earlier maximum. |
| 3 | `(0.464942, 0.606056, 0.448490)` | Refine the neighbourhood of the Week 4 maximum with moderate exploration. |
| 4 | `(0.395750, 0.355914, 0.357177, 0.393665)` | Test a constrained candidate supported by the better-performing GP models. |
| 5 | `(0.950000, 1.000000, 1.000000, 1.000000)` | Test whether reducing the first input has an effect comparable to the Week 8 and 9 tests of the second and third inputs. |
| 6 | `(0.391479, 0.366644, 0.681207, 0.766389, 0.083007)` | Continue exploring around the new Week 9 maximum. |
| 7 | `(0.016725, 0.215301, 0.249637, 0.368315, 0.280830, 0.630035)` | Refine the improved Week 9 region using an RBF/Matérn ensemble. |
| 8 | `(0.121616, 0.212254, 0.127558, 0.105313, 0.720775, 0.503004, 0.199956, 0.694395)` | Test a broader direction within the promising region using the GP ensemble. |

The Function 5 query is a **deliberate controlled test**, rather than the unmodified optimum of its acquisition function. For Function 1, I likewise prioritised better spatial coverage over a small advantage in predicted UCB.

**Week 10 results are not included here.** The latest evaluated dataset contains 247 observations: 175 initial observations and 72 outputs returned from Weeks 1–9. The eight Week 10 selections above should not be counted as completed evaluations until their outputs are available. The latest best observed results and surrogate performance comparisons are in the [model card](model_card.md).

## 4. Lessons and next steps

The main change in my approach was moving from a fairly standard GP-UCB procedure towards **function-specific, evidence-led query selection**. I found that an apparently accurate model could still recommend an unproductive query, while a less accurate prediction might lead to a new maximum. LOOCV, chronological backtesting and comparisons between kernel models helped me understand those trade-offs, although none removed the uncertainty created by small and adaptively collected datasets.

Once the Week 10 outputs are available, I will compare them with the recorded predictions, update the best observed values and reflect on which final-round decisions were most useful. The [model card](model_card.md) will hold the detailed performance assessment, and the [datasheet](data_sheet.md) will be updated to include any new evaluated observations.
