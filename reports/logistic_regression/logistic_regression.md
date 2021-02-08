Logistic Regression model
==============================

The first model to be implemented was a logistic regression. It seems natural, as the target variable is binary.


Characteristics
------------

The optimal hyperparameters were the following.

| Hyperparameter        | Tested range     | Optimal value     |
| :------------- | :----------: | :-----------: |
|  penalty | {'l1', 'l2'}   | 'l2' |
|  C | [1e-4, 1e10]   | 10 |
|  fit_intercept | {True, False}   | True |
|  n\_features\_to\_select | [0,55]   | 11 |

Feature Selection is performed with the Recursive Feature Elimination strategy proposed by `scikit-learn`. It removes features recursively until the desired number of features.

As we already perform feature selection, the L2-norm for regularization will be more suited than the L1-norm.

Principal Component Analysis was tested but underperformed.

SMOTE oversampling was tested but underperformed.


Hyperparameters
------------

**penalty**: Regularization. L1 or L2 is usable with the `liblinear` and `saga` solvers.

**C**: Penalty term for the regularization. It is the inverse of the lambda regularization parameter. $ C = \frac{1}{\lambda} $
The higher the value, the less we penalize 

**fit\_intercept**: Whether a constant should be added to the model.

**n\_features\_to\_select**: The number of features used in the model, selected during the Recursive Feature Selection.

### Implementation

Hyperparameter search was performed using the Grid Search function with cross-validation (5 folds), proposed with `GridSearchCV` in the `scikit-learn` module.

