# Dataset Datasheet: Black-Box Optimisation Capstone

**Professional Certificate in Machine Learning and Artificial Intelligence**  
**Imperial College Business School, London**  
**Dataset status:** Results recorded through Week 9 (28 October 2026)

## Motivation

This dataset was developed as part of my capstone project for the Professional Certificate in Machine Learning and Artificial Intelligence, offered by Imperial College Business School in London.

The project is based on the approach used in the NeurIPS 2020 Black-Box Optimization for Machine Learning competition. The aim is to find the highest possible output from eight unknown functions, using a limited number of evaluations. For each function, I can choose an input and observe the resulting output, but I do not know the mathematical expression or how the function behaves elsewhere.

I used the observations to build surrogate models, identify promising inputs and decide when to explore unfamiliar regions rather than refine areas that had already performed well. The main purpose of the dataset is to support and assess those decisions, not to train a conventional supervised model for deployment.

The initial observations and access to the functions were provided through the programme. I selected the subsequent inputs and collected their outputs through the capstone evaluation platform. I do not have details of the original function designers or any separate funding arrangements.

## Composition

Each observation consists of a numerical input vector and the single output returned by one of the eight functions. The functions have between two and eight input dimensions, with each input coordinate ranging from 0 to 1. Each function is treated as a separate optimisation problem, since their output scales differ considerably.

| Function | Input dimensions | Initial observations | Further observations (Weeks 1–9) | Total |
|---|---:|---:|---:|---:|
| 1 | 2 | 10 | 9 | 19 |
| 2 | 2 | 10 | 9 | 19 |
| 3 | 3 | 15 | 9 | 24 |
| 4 | 4 | 30 | 9 | 39 |
| 5 | 4 | 20 | 9 | 29 |
| 6 | 5 | 20 | 9 | 29 |
| 7 | 6 | 30 | 9 | 39 |
| 8 | 8 | 40 | 9 | 49 |
| **Total** | | **175** | **72** | **247** |

The dataset contains 247 evaluated input-output pairs through Week 9. I checked that the recorded inputs and outputs were complete and correctly matched. No missing or non-finite numerical values or repeated input vectors were identified within any function.

The observations contain numerical values rather than information about individuals or organisations, and no personal or confidential information is apparent. However, the underlying challenge materials may still be subject to the programme's terms of use.

## Collection process

The project began with an initial set of observations for each function. I do not know the sampling method used to choose these starting points.

I then submitted one new input for each function in each of nine weekly rounds. The evaluation platform returned the corresponding outputs, which I used to update my models and select the next round of inputs. My approach evolved during the project: early queries involved broader searches, while later queries increasingly used Gaussian Process predictions, acquisition functions and local refinement around promising regions.

The later observations are therefore not a random sample of the full input space. They reflect decisions made using the results available at the time, and some parts of the search space are much more heavily sampled than others.

This datasheet covers completed evaluations through Week 9. Proposed Week 10 inputs are not counted as observations because their outputs had not yet been returned at this stage.

## Preprocessing, cleaning and labelling

The input values and returned outputs were retained as numerical observations. No class labels were needed, and I did not remove or impute any observations.

For modelling, I combined the initial observations with the weekly results for each function and organised them into input matrices and output vectors. I used target normalisation within several Gaussian Process models to make fitting more stable. For Function 5, I also tested a square-root transformation of the non-negative outputs because of their relatively large range. The original objective values were retained for reporting and comparing optimisation results.

Before fitting the models, I checked that the observations were consistent. The capstone required every submitted input coordinate to be given to six decimal places, so I used those submitted coordinates when pairing weekly queries with their returned outputs. The initial observations were retained at their original precision. I assessed model performance using leave-one-out cross-validation and chronological backtesting, focusing on how well earlier observations predicted later results.

## Uses

The primary use of this dataset is to study black-box optimisation under a limited evaluation budget. It allows me to compare surrogate models and acquisition strategies, assess prediction errors and uncertainty, and examine the trade-off between exploration and exploitation.

There are several limitations to its wider use. The number of observations is small relative to the dimensions of some functions. Later observations are concentrated in areas that earlier models identified as promising, so performance on these data may not reflect performance across the full search space. In addition, the underlying functions are unknown: the highest observed output should not be treated as proof that the global maximum has been found.

The dataset could also be useful for comparing other optimisation methods, provided any such comparison accounts for the sequential way the observations were collected. It should not be treated as a representative real-world dataset or used to draw conclusions about causal relationships or outcomes involving people or businesses.

## Distribution

The observations were collected for an educational capstone and used in developing my optimisation approach. I intend to document the methodology and results in my GitHub portfolio. I have not established whether the programme permits public redistribution of the underlying challenge observations, so their release would require confirmation of the relevant terms.

## Maintenance

I maintain the dataset and the documentation for my capstone project. As further results become available, I will add the evaluated input-output pairs, retain earlier observations and update the reported totals. I will also distinguish actual results from model predictions so that the development of the optimisation strategy remains clear.
