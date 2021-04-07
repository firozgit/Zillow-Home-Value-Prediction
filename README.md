Zillow Home Value Prediction
==============================

This Project demonstrates the data wrangling, data exploration, feature engineering and predictive modeling process on the Kaggle's Zillow Home Value dataset (https://www.kaggle.com/c/zillow-prize-1/overview)

Zillow's **Zestimate** are the estimated home values based on 7.5 million statistical and machine learning models that analyze hundreds of data points on each property. In the first round of the Kaggle competition, users have to predict the Zestimate residual error and submissions are evaluated on **Mean Absolute Error** between the predicted log error and the actual log error.

The log error is defined as **logerror=log(Zestimate)−log(SalePrice)**

Throughout the notebooks in this repository, we will use the given property data and transaction data to predict the logerror. 

Some of the functions and classes are written in forms of scripts that can be reused whenever required. 




Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── data_processor.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── tune_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations   


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
