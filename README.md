# Black-Box Optimisation Capstone Project

## 1. Project Overview

This project explores the application of machine learning techniques to black-box optimisation (BBO), where the objective is to identify optimal inputs for unknown functions using a limited number of evaluations.

Unlike conventional optimisation problems, the underlying mathematical expressions, gradients and response surfaces are unavailable. Instead, optimisation relies on observed input-output relationships to progressively identify promising regions.

The capstone involves eight unknown continuous functions with input dimensionalities ranging from two to eight. Each function presents different challenges, including nonlinear relationships, potential local optima and varying sensitivity to input variables.

My primary approach involves **Gaussian Process Regression (GPR) and Bayesian optimisation**, which provide a probabilistic framework for predicting function values and quantifying uncertainty.

These techniques are relevant to real-world machine learning applications, including hyperparameter optimisation, experimental design and resource allocation, where obtaining additional observations can be costly.

The project also supports my professional development by strengthening my ability to apply data science techniques to decision-making problems involving incomplete information and limited resources.

## 2. Inputs and Outputs

Each unknown function accepts a continuous input vector:

$$
X = (x_1, x_2, \ldots, x_d)
$$

where the dimensionality $d$ ranges from 2 to 8, and each input is constrained to:

$$
0 \leq x_i \leq 1
$$


For example, a valid three-dimensional query might be:

```python
X = [0.4, 0.6, 0.2]
```

Each query returns a scalar numerical output:

```python
y = f(X)
```

The output represents the objective value associated with the submitted input. The underlying relationship between inputs and outputs remains unknown.

Historical observations are accumulated into datasets containing evaluated input coordinates and their corresponding outputs. These provide the training data for the surrogate models.

## 3. Challenge Objectives

The primary objective is to **maximise each unknown function** by identifying inputs that produce the highest possible output values.

The main challenges include:

- Limited function evaluations, making each query valuable.
- Unknown function structures and potentially complex response surfaces.
- Increasing dimensionality, which makes searching the input space more difficult.
- Uncertainty in predictions, particularly in unexplored regions.

A successful query may either improve the best observed value or provide information that enables more effective future decisions.

Consequently, optimisation performance depends on balancing **exploration**, which investigates uncertain regions, and **exploitation**, which focuses on areas with promising predicted outputs.

## 4. Technical Approach and Strategy Evolution

### Week 1: Establishing a Baseline

My initial strategy involved fitting Gaussian Process models to the available observations.

GPR was selected because it provides both predictions of unknown function values and estimates of predictive uncertainty.

I used the Upper Confidence Bound (UCB) acquisition function to identify candidate queries based on a combination of predicted performance and uncertainty.

This established a systematic alternative to purely random search.

### Week 2: Refining the Surrogate Model

Following the first additional observations, I began refining the modelling assumptions and considering how GP kernel hyperparameters, particularly length scales, affect predictions.

I also examined how adjusting the UCB exploration parameter could shift query selection between promising existing regions and less-explored areas.

This reinforced the importance of adapting the optimisation strategy as new information becomes available.

### Week 3: Balancing Global and Local Exploration

By the third iteration, I increasingly considered differences between the eight functions rather than relying on identical optimisation settings.

I explored the potential benefits of different kernel configurations and considered whether queries should focus on local refinement or wider exploration.

The objective was to identify promising regions without prematurely excluding areas that might contain better solutions.

## 5. Future Development

As additional observations become available, I intend to investigate more systematic hyperparameter tuning, alternative covariance kernels and methods for evaluating predictive accuracy.

I also plan to explore whether local optimisation strategies or search-space partitioning could improve efficiency, particularly for higher-dimensional functions.

Ultimately, this project aims to demonstrate how probabilistic modelling and iterative experimentation can support effective optimisation when the underlying problem is uncertain and evaluation opportunities are limited.
