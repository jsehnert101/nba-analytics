nba-analytics
==============================

Wrangle and analyze NBA statistics and betting information.

------------

Prerequisites

The environment configuration relies on an [Anaconda Distribution](https://anaconda.org). Foe example, I use micromamba on my Apple Silicon device. 
You should be familiar with the following in order to use this package:

- git
- python
- conda
- pandas
- jupyter notebook
- [nba-api](https://github.com/swar/nba_api/tree/master)
- [odds-api](https://the-odds-api.com)
- **Bonus:** requests

Directions
------------

```bash
# Clone the repository, ideally via ssh like:
git clone git@github.com:jsehnert101/nba-analytics.git

# Create conda env with required packages
conda env create -f environment.yml

# Activate conda environment
conda activate nba

# Install local packages
pip install -e ./src

# Test environment configuration
python test_environment.py

# Get coding!
# If you're looking for ways to get started, check our example notebooks under docs/examples.
```

Refer to the notebooks in docs/examples for more information on retrieving and storing data locally, computing advanced statistics, applying machine learning models, and more!

Please read this if you plan on contributing.
------------

This project organization allows us to effectively build out this package over time. All exploratory work should go inside a relevant subfolder within the eda (exploratory data analysis) folder. If no appropriate category for you work exists, create it. Please do not push to the src folder without a code review. If you plan on making substantial changes, create a branch and request approval from Jake Sehnert prior to merge.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
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
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
