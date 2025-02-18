# ML_project

[![Run Python Tests](https://github.com/AndreCanto00/ML_project/actions/workflows/test.yml/badge.svg)](https://github.com/AndreCanto00/ML_project/actions/workflows/test.yml)

## Project:

The Research & Development team of Tyre Inc. has been testing a number of new products
for potential introduction to the market. For each tyre, a number of attributes have been collected
about the tyre characteristics and the testing conditions, as well as the outcome of the test, i.e.,
success or failure. As member of the team in charge of data science you are in charge to create a
machine learning task to predict the failure of a tyre and describe which attributes are the most
relevant for outcome.

**Dataset description**: The dataset contains a list of tested tyres. The target and attributes
information are described below.
  • Number of instances: 3000
  • number of attributes: 15
  • Target variable: “failure”:
    – 0: test was successful
    – 1: test ended with tyre failure
    
The task is formulated as a binary classification. Your grade will be based on the F1-score
metric and on the modeling process presented in the report.

### Description of Folders and Files.

- `.github/workflows/`: Contains GitHub Actions workflows for continuous integration, such as the `test.yml` file for running tests.
- `.gitignore`: File to specify which files and folders should be ignored by Git.
- `.vscode/`: Contains Visual Studio Code-specific settings, such as `settings.json`.
- `10681109_Cantore_Andrea-2.ipynb`: Jupyter notebook for project analysis and development.
- `Makefile`: File to automate build and project management operations.
- `README.md`: This file, which contains the project documentation.
- `requirements.txt`: List of Python dependencies needed for the project.
- `scaler.pkl`: Pickle file containing the scaler used for data normalization.
- `src/`: Folder containing the source code for the project.
  - `__init__.py`: File to make `src` a Python module.
  - `data_preparation.py`: Script for preparing data.
  - `data_processing.py`: Script for data processing.
  - `exploratory_analysis.py`: Script for exploratory data analysis.
  - `visualization.py`: Script for data visualization.
- `tests/`: Folder containing unit tests for the project.
  - `__init__.py`: File for making `tests` a Python module.
  - `test_data_preparation.py`: Tests for `data_preparation.py`.
  - `test_data_processing.py`: Test per `data_processing.py`.
  - `test_exploratory_analysis.py`: Test per `exploratory_analysis.py`.
  - `test_visualization.py`: Test per `visualizzazione.py`.
- `tyres_test.csv`: Dataset di test.
- `tyres_train.csv`: Dataset di addestramento.