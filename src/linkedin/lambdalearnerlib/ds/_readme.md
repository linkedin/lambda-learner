# Data Structures (DS) package

This module contains code defining the internal data structures used by lambda to store training data and coefficients, as well as utilities for converting between various representations of this data.

Broadly there are two types of representations:

1. Name-term-value. This is the form in which the data and coefficients are read and written.
2. Index-value. This is the form required by training and scoring.

We convert between these two representations using an index map which is also included in this module.

## Dependencies

Code in this package can depend on:

- `samza.ai.utils`