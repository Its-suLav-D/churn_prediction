# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project is part of the ML DevOps Engineer Nanodegree program at Udacity. It focuses on identifying credit card customers that are most likely to churn. The project includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested).

The project involves the following main steps:
1. Data import and exploratory data analysis (EDA)
2. Feature engineering and data preprocessing
3. Model training (Random Forest and Logistic Regression)
4. Model prediction and evaluation
5. Result visualization

## Files and Data Description
The project contains the following main files:

- `churn_library.py`: The main Python script containing functions to run the customer churn prediction pipeline.
- `churn_script_logging_and_tests.py`: A script for testing the functions in `churn_library.py` and logging the results.
- `README.md`: This file, providing an overview of the project.

Data:
- `data/bank_data.csv`: The dataset used for this project. It contains customer information and churn status.

Directories:
- `images/`: Contains EDA results and model performance visualizations.
  - `eda/`: Exploratory Data Analysis plots
  - `results/`: Model performance plots
- `models/`: Stores the trained machine learning models.
- `logs/`: Contains log files generated during script execution.

## Installation
To set up the project environment:

1. Clone this repository to your local machine.
2. Ensure you have Anaconda or Miniconda installed on your system.
3. Create a new Conda environment using the provided `environment.yml` file:
   ```
   conda env create -f environment.yml
   ```
   This will create a new Conda environment named `churn_prediction` with all the required dependencies.

4. Activate the new environment:
   ```
   conda activate churn_prediction
   ```

5. Verify that the environment was installed correctly:
   ```
   conda env list
   ```

## Running Files
To run the customer churn prediction pipeline:

1. Ensure you're in the project's root directory.
2. Run the main script:
   ```
   python churn_library.py
   ```
   This will perform the entire pipeline: data import, EDA, feature engineering, model training, and result generation.

3. To run tests and generate logs:
   ```
   python churn_script_logging_and_tests.py
   ```
   This will test each function in `churn_library.py` and log the results in `./logs/churn_library.log`.

## Expected Outputs
After running `churn_library.py`:
- EDA plots will be saved in `./images/eda/`
- Trained models will be saved in `./models/`
- Model performance plots will be saved in `./images/results/`

After running `churn_script_logging_and_tests.py`:
- Test results and any error messages will be logged in `./logs/churn_library.log`
