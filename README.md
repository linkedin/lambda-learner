# Lambda Learner

## What is it
Lambda Learner is a framework for training models by incremental updates in response to mini-batches from data streams.


## Why Lambda Learner
One of the most well-established applications of machine learning is in deciding what content to show website visitors. When observation data comes from high-velocity, user-generated data streams, machine learning methods perform a balancing act between model complexity, training time, and computational costs. Furthermore, when model freshness is critical, the training of models becomes time-constrained. Parallelized batch offline training, although horizontally scalable, is often not time-considerate or costeffective.

Lambda Learner is capable of incrementally training the memorization part of the model as a booster over the generalization part. The frequent updates it brings can improve business metrics.

## How to use Lambda Learner
### Installation
```python

pip install labmbda-learner
```

### Training with Lambda Learner

```python

# Load in and prepare your data and current model (a specific random effect), formatted using "Name Term Value" features.
training_data: List[TrainingRecord] = ...
test_data: List[TrainingRecord] = ...
model_coefficients: List[Feature] = ...
model_coefficienet_variances: List[Feature] = ...

# Create an index map supporting two way mapping of all features in the training data and the model.
# `index_map_metadata` contains index map statistics, which be logged or used when debugging.
index_map, index_map_metadata = IndexMap.from_records_means_and_variances(
	training_data, model_coefficients, model_coefficienet_variances)

# Vectorize the training data
indexed_training_data = nt_domain_data_to_index_domain_data(training_data, index_map)

regularization_penalty = 10.0

# Vectorize the model
initial_model = nt_domain_coeffs_to_index_domain_coeffs(model_coefficients, model_coefficienet_variances, index_map, regularization_penalty)

forgetting_factor = 0.8

# Perform training, using the Lambda Learner Sequential Bayesian update loss.
# Two other trainers are also currently available: TrainerLogisticLossWithL2 and TrainerSquareLossWithL2.
lr_trainer = TrainerSequentialBayesianLogisticLossWithL2(
    training_data=indexed_training_data,
    initial_model=initial_model,
    penalty=regularization_penalty,
    delta=forgetting_factor)

# `training_metadata` contains the metadata returned by scipy fmin_l_bfgs_b optimizer, which be logged or used when debugging.
# Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
updated_model, updated_model_loss, training_metadata = lr_trainer.train()

# Score the updated model
indexed_test_data = nt_domain_data_to_index_domain_data(test_data, index_map)
post_train_scores = score_linear_model(updated_model, indexed_test_data)

# Evaluate the updated model. Currently 'auc' and 'rmse' metrics are supported.
trained_model_metrics = evaluate(metric_list=['auc'], y_scores=post_train_scores, y_targets=training_data.y)
trained_model_auc = post_train_metrics['auc']

# Convert model back to Name Term Value domain.
mean_dict, var_dict = index_domain_coeffs_to_nt_domain_coeffs(updated_model, index_map)

# Save this random effect model, ready for the next incremental update.
```

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License
This project is licensed under the BSD 2-CLAUSE LICENSE - see the [LICENSE](LICENSE) file for details.
