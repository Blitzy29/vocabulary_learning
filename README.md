Vocabulary Learning
==============================

This project is dedicated to learn words in a new language (German in my case).

Process
------------

The principle is simple:

* a word in English or German is proposed to the user, who has to answer with its translation
* user enters her answer
* a message is prompted to the user
	* if the translation was correct, it congratulates you
	* if not, the correct word is prompted ; if you confused it with another word, it tells you ; a text box also opens to write the word again

<p align="center">
<img src="reports/example.png" width="500">
</p>


All translations are recorded, with information like the language asked, the time, or how many words were already translated.

A model is then build before a next session to predict whether a word is already known by the user. These predictions are later used when choosing a word: a word with a high probability of being known has a lower chance of being asked than a word of lower probability.


Models
------------

2 models are currently implemented.

1. A Logistic regression
2. A Gradient Boosting algorithm


A Luigi pipeline takes care of the creation process.


### Model performances

| Metric        | Logistic Regression     | Gradient Boosting     |
| :------------- | :----------: | :-----------: |
|  Precison | <font color="red">0.88</font>   | <font color="green">0.97</font>    |
| Recall   | <font color="red">0.81</font> | <font color="green">0.91</font> |
| F1-score   | <font color="red">0.84</font> | <font color="green">0.94</font> |
| Accuracy   | <font color="red">0.78</font> | <font color="green">0.92</font> |
| ROC AUC   | <font color="red">0.76</font> | <font color="green">0.92</font> |
| MAE   | <font color="red">0.30</font> | <font color="green">0.16</font> |


<p align="center">
<img src="reports/20210119_comparison_roc_auc_curve.png" width="700">
</p>


Visualization
------------

##### Prediction labels

This plot represents the predicted datapoints on the test dataset. On the x-axis are represented datapoints, sorted by predicted value (from 0 to 1) ; on the y-axis their values. In grey are the predicted value. True values are plotted on y=1 (if positive) and on y=0 (if negative), with 2 colors: in green if they were correctly predicted, on red if not. The diagonal dashed line represents the change between negative and positive values in the true values.

<p align="center">
<img src="reports/dataviz_prediction_labels__1.png" width="300">
<img src="reports/dataviz_prediction_labels__2.png" width="300">
</p>

On the first graph, we observe a (too) smooth transition in the predicted values between positive and negative. No clear distinction is made, and we have lots of predicted values between 0.3 and 0.8. We also observe a bump around 0.2.

On the second graph, in this case, a clear distinction is made between negative and positive. However we can observe that the model is unbalanced: we predict more negative values than there really is.

#####  Distribution of the predictions before learning

These graphs represent the distributions of the predictions for English and German before a learning session. On the x-axis and top are represented the German predictions, and the y-axis and right are represented the English predictions.

<p align="center">
<img src="reports/dataviz_distribution_prediction__1.png" width="300">
<img src="reports/dataviz_distribution_prediction__2.png" width="300">
</p>

On the first graph, we can see that we are before a session where the user has a lot of unknown words, and is in the process of learning the others. 

Distributions are nearly symmetrical. Drawing the diagonal line would show us that the English predictions are slightly higher than German predictions, which is expected: it is easier for me to say the English translation of a German word, than the German translation of an English word.

On the second graph, we observe that most of the words are known: the distributions are concentrated on 1. Other words are in the process of being known.


Next steps
------------

- [ ] New model: Recurrent Neural Network, using binary {known_unknown} variables as training data


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── log            <- Logs using a default name: 'YYYYmmdd_HHMMSS_file_handler_name'
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── unit_tests     <- unit tests
    │   │
    │   ├── utils          <- utilities, functions which can safely be used in all programs
    │   │   └── io.py
    │   │   └── logger_module.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
