"""
This module contains unit tests for the churn_library module.

Author: Sulove Dahal
Date: 2024-10-15 
"""

import os
import logging
import numpy as np
import pandas as pd
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
    '''
    Test data import
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_eda(perform_eda):
    '''
    Test perform eda function
    '''
    df = cls.import_data("./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    try:
        perform_eda(df)
        assert os.path.isfile('./images/eda/churn_distribution.png')
        assert os.path.isfile('./images/eda/customer_age_distribution.png')
        assert os.path.isfile('./images/eda/marital_status_distribution.png')
        assert os.path.isfile('./images/eda/total_transaction_distribution.png')
        assert os.path.isfile('./images/eda/heatmap.png')
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: EDA images are missing")
        raise err

def test_encoder_helper(encoder_helper):
    '''
    Test encoder helper
    '''
    df = cls.import_data("./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    
    try:
        df_encoded = encoder_helper(df, cat_columns, 'Churn')
        for col in cat_columns:
            assert f'{col}_Churn' in df_encoded.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: Encoded columns are missing")
        raise err

def test_perform_feature_engineering(perform_feature_engineering):
    '''
    Test perform_feature_engineering
    '''
    df = cls.import_data("./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    df = cls.encoder_helper(df, cat_columns, 'Churn')
    
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: Output data shapes or types are incorrect")
        raise err

def test_train_models(train_models):
    '''
    Test train_models function
    '''
    df = cls.import_data("./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    df = cls.encoder_helper(df, cat_columns, 'Churn')
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, 'Churn')
    
    try:
        train_models(X_train, X_test, y_train, y_test)
        assert os.path.isfile('./models/rfc_model.pkl')
        assert os.path.isfile('./models/logistic_model.pkl')
        assert os.path.isfile('./images/results/random_forest_results.png')
        assert os.path.isfile('./images/results/logistic_regression_results.png')
        assert os.path.isfile('./images/results/roc_curve_result.png')
        assert os.path.isfile('./images/results/feature_importances.png')
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models: Model files or results images are missing")
        raise err

def test_feature_importance_plot(feature_importance_plot):
    '''
    Test feature_importance_plot function
    '''
    df = cls.import_data("./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    df = cls.encoder_helper(df, cat_columns, 'Churn')
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, 'Churn')
    

    rfc = cls.RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)
    
    try:
        feature_importance_plot(rfc, X_train, './images/results/test_feature_importance.png')
        assert os.path.isfile('./images/results/test_feature_importance.png')
        logging.info("Testing feature_importance_plot: SUCCESS")
    except AssertionError as err:
        logging.error("Testing feature_importance_plot: Feature importance plot is missing")
        raise err

def test_classification_report_image(classification_report_image):
    '''
    Test classification_report_image function
    '''
    df = cls.import_data("./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    df = cls.encoder_helper(df, cat_columns, 'Churn')
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, 'Churn')
    
    rfc = cls.RandomForestClassifier(random_state=42)
    lrc = cls.LogisticRegression(random_state=42)
    
    rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)
    
    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    
    try:
        classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)
        assert os.path.isfile('./images/results/random_forest_results.png')
        assert os.path.isfile('./images/results/logistic_regression_results.png')
        logging.info("Testing classification_report_image: SUCCESS")
    except AssertionError as err:
        logging.error("Testing classification_report_image: Classification report images are missing")
        raise err

if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
    test_feature_importance_plot(cls.feature_importance_plot)
    test_classification_report_image(cls.classification_report_image)