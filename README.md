# Lambda Learner

## What is it

Lambda Learner is a library for iterative incremental training of a class of supervised machine learning models. Using the Generalized Additive Mixed-Effect (GAME) framework, one can divide a model into two components, (a) Fixed Effects - a typically large "fixed effects" model (generalization) that is trained on the whole dataset to improve the model’s performance on previously unseen user-item pairs, and (b) Random Effects - a series of simpler linear "random-effects" models (memorization) trained on data corresponding to each entity (e.g. user or article or ad) for more granular personalization.

The two main choices in defining a GAME architecture are 1) choosing the model class for the fixed effects model, and 2) choosing which random effects to include. The fixed effects model can be of any model class, typically [Tensorflow](https://github.com/tensorflow/tensorflow), [DeText](https://github.com/linkedin/detext), [GDMix](https://github.com/linkedin/gdmix), [XGBoost](https://github.com/dmlc/xgboost). As for the random effects, this choice is framed by your training data; specifically by the keys/ids of your training examples. If your training examples are keyed by a single id space (say userId), then you will have one series of random effects keyed by userId (per-user random effects). If your data is keyed by multiple id spaces (say userId, movieId), then you can have up to one series of random effects for every id type (per-user random effects, and per-movie random effects). However it's not necessary to have random effects for all ids, with the choice being largely a modeling concern.

Lambda Learner currently supports using any fixed-effects model, but only random effects for a single id type.

Bringing these two pieces together, the residual score from the fixed effects model is improved using a random effect linear model, with the global model's output score acting as the bias/offset for the linear model. Once the fixed effects model has been trained, the training of random effects can occur independently and in parallel. The library supports incremental updates to the random effects components of a GAME model in response to mini-batches from data streams. Currently the following algorithms for updating a random effect are supported:

- Linear regression.
- Logistic regression.
- Sequential Bayesian logistic regression (as described in the [Lambda Learner paper](https://arxiv.org/abs/2010.05154)).

The library supports maintaining a model coefficient Hessian matrix, representing uncertainty about model coefficient values, in addition to point estimates of the coefficients. This allows us to use the random effects as a multi-armed bandit using techniques such as Thompson Sampling.


## Why Lambda Learner

One of the most well-established applications of machine learning is in deciding what content to show website visitors. When observation data comes from high-velocity, user-generated data streams, machine learning methods perform a balancing act between model complexity, training time, and computational costs. Furthermore, when model freshness is critical, the training of models becomes time-constrained. Parallelized batch offline training, although horizontally scalable, is often not time-considerate or cost effective.

Lambda Learner is capable of incrementally training the memorization part of the model (the random-effects components) as a performance booster over the generalization part. The frequent updates to these booster models over already powerful fixed-effect models improve personalization. Additionally, it allows for applications that require online bandits that are updated quickly.

In the GAME paradigm, random effects components can be trained independently of each other. This means that their update can be easily parallelized across nodes in a distributed computation framework. For example, this library can be used on top of Python Beam or PySpark. The distributed compute framework is used for parallelization and data orchestration, while the Lambda Learner library implements the update of random effects in individual compute tasks (DoFns in Beam or Task closures in PySpark).


## Installation

```bash
pip install lambda-learner
```

## Tutorial: How to use Lambda Learner

### Prepare your dataset and initial model

Let's assume we have a minibatch of data, a random effects model for a specific key, and the already trained global fixed effects model. In order to use Lambda Learner, we need to format the data and model into appropriate data structures as follows:

```python
training_data: List[TrainingRecord] = ...
test_data: List[TrainingRecord] = ...
model_coefficients: List[Feature] = ...
model_coefficienet_variances: List[Feature] = ...
```

A `TrainingRecord` represents a labeled example. The most important fields in this structure are:

- `label` => The datum label. For example, this could be binarized (0.0, or 1.0) for a classification task, or in the range [0.0,1.0] for a regression task.
- `features` => A list of `Feature`s. `Feature` is a Name-Term-Value representation which we'll discuss next.
- `offset` => The score that the associated fixed-effects model produces for this datum. The score from a deep or non-linear fixed-effect model is captured in just one parameter. We use this score as the residual to train the random-effect models.

Both features (from the training data) and model coefficients are represented using the `Feature` class. `Feature` is a Name-Term-Value (NTV) representation, where the name is the feature name, the term is a string index for the feature (supporting categorical and numerical vector features), and the value is the numerical value corresponding to a name-term pair. When a `Feature` is used to describe a model, the value is the coefficient weight.

Here's a toy example of data and a model using single feature: a categorical representing a user's favorite season of the year. In actual practice, you would create these data structures by reading in external resources and wrangling them into this form.

```python
training_data = [
	TrainingRecord(
		label=1.0,
		weight=1.0,
		offset=0.6786987785,  # determined by scoring this example using your global model.
		features=[
			# one feature with multiple terms, a categorical vector
			Feature("season", "winter", 1.0),
			Feature("season", "spring", 0.0),
			Feature("season", "summer", 0.0),
			Feature("season", "fall", 0.0)
		]
	),
	# more records...
]

model_coefficients = [
	# All models need an intercept feature, corresponding to the `offset` field in the data.
	Feature("intercept", "intercept", 1.0),

	# one feature with multiple terms, a categorical vector
	Feature("season", "winter", 0.423),
	Feature("season", "spring", 0.564),
	Feature("season", "summer", 0.234),
	Feature("season", "fall", 0.0344)
]
```

In the future, other storage formats besides NTV may be supported.

### Create an index map

NTV is a very human-readable format for representing the model coefficients and data record features. However, in order to train the model, we need to transform both the model data into an indexed, vector representation. An `IndexMap` is a (bi-directional) mapping between a Name-Term and an integer index, which we use to translate from the human-readable NTV representation to an trainable indexed representation.

```python
index_map, index_map_metadata = IndexMap.from_records_means_and_variances(
	training_data, model_coefficients, model_coefficienet_variances)
```

`index_map_metadata` contains index map statistics, which can be logged or used for monitoring.

### Transform your model and data into an indexed representation

Now that we have an `index_map`, we can use helper functions from `representation_utils.py` to transform our data and model from NTV-representations to indexed representations, as follows:

```python
indexed_training_data = nt_domain_data_to_index_domain_data(training_data, index_map)
indexed_test_data = nt_domain_data_to_index_domain_data(test_data, index_map)

regularization_penalty = 10.0
initial_model = nt_domain_coeffs_to_index_domain_coeffs(model_coefficients, model_coefficienet_variances, index_map, regularization_penalty)
```

The data and model are now ready for training.

### Perform training

To perform training, choose one of the `Trainer` subclasses appropriate for your task:

- `TrainerSquareLossWithL2` for linear regression.
- `TrainerLogisticLossWithL2` or `TrainerSequentialBayesianLogisticLossWithL2` for classification.

```python
forgetting_factor = 0.8
lr_trainer = TrainerSequentialBayesianLogisticLossWithL2(
    training_data=indexed_training_data,
    initial_model=initial_model,
    penalty=regularization_penalty,
    delta=forgetting_factor)

updated_model, updated_model_loss, training_metadata = lr_trainer.train()
```

`training_metadata` contains the metadata returned by the scipy `fmin_l_bfgs_b` optimizer, which be logged or used when debugging. See [Scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html) for more information.

`updated_model` is an `IndexedModel` which is the result or this minibatch training iteration.


### Perform scoring and metric evaluation

Next we'll score our test set using `updated_model`, and evaluate the model's performance. `evaluate` can compute several metrics in one go, but here we request just AUC ([area under the ROC curve](https://www.wikiwand.com/en/Receiver_operating_characteristic#/Area_under_the_curve)), a common binary classification metric.

```python
scores = score_linear_model(updated_model, indexed_test_data)
trained_model_metrics = evaluate(metric_list=['auc'], y_scores=scores, y_targets=indexed_test_data.y)
trained_model_auc = post_train_metrics['auc']
```

### Transform your data back into a human readable representation

Finally, we transform our model back into a NTV representation using another helper from `representation_utils.py`.

```python
means, variances = index_domain_coeffs_to_nt_domain_coeffs(updated_model, index_map)
```

`means` and `variances` represent the updated model coefficients and their variances. These can now be stored and subsequently used for inference or further updated on the next data minibatch.

## Citing

Please cite Lambda Learner in your publications if it helps your research:

```
@misc{ramanath2020lambda,
      title={Lambda Learner: Fast Incremental Learning on Data Streams},
      author={Rohan Ramanath and Konstantin Salomatin and Jeffrey D. Gee and Kirill Talanine and Onkar Dalal and Gungor Polatkan and Sara Smoot and Deepak Kumar},
      year={2020},
      eprint={2010.05154},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.


## License

This project is licensed under the BSD 2-CLAUSE LICENSE - see the [LICENSE](LICENSE) file for details.
