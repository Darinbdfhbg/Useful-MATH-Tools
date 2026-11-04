# Linear Approximation & Error Analysis

## Overview

This project is a Python-based mathematical tool that explores linear approximation and its accuracy.

It uses symbolic differentiation, numerical calculations, and visualization to analyze how well a tangent line approximates a function near a given point.

## Features

- Supports different functions your given.
- Gets derivatives at the same time (SymPy).
- Computes linear approximations using the tangent line formula.
- Calculates absolute and relative errors.
- Determines whether the approximation meets a user-defined error tolerance.
- Numerically estimates the interval in which the relative error meets the tolerance.
- Generates PNG visualizations of the original function, tangent line, and estimated valid interval.

## Mathematical Background

The linear approximation of a differentiable function near a point \(a\) is:

\[
L(x)=f(a)+f'(a)(x-a)
\]

The relative error is calculated by:

\[
E_r=\frac{|L(x)-f(x)|}{|f(x)|}\times100\%
\]

The estimated valid interval is obtained using numerical sampling and the bisection method. This numerical estimate is not a rigorous mathematical guarantee.

## Technologies

- Python
- SymPy
- NumPy
- Matplotlib

## Example

For the function \(f(x)=x^2\), with expansion point \(a=2\), the linear approximation is:

\[
L(x)=4x-4
\]

With a relative error tolerance of 1%, the estimated valid interval near \(a=2\) is approximately:

\[
[1.818182,\ 2.222222]
\]

## Limitations

The program supports many common real-valued functions, but not all mathematical functions. Relative error is undefined where the true function value is zero. The valid interval is estimated numerically and may be inaccurate for highly oscillatory or discontinuous functions.

## Project Status

Version 2.2 — Linear Approximation and Valid Interval Estimation.

This project is part of the Useful-MATH-Tools collection.
