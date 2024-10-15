# library doc string
"""
This module contains functions for the customer churn prediction project.

Author: Sulove Dahal
Date: 2024-10-15 
"""

# Import libraries
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.preprocessing import StandardScaler

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Constants
KEEP_COLS = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
    'Income_Category_Churn', 'Card_Category_Churn'
]


def import_data(pth):
    """
    Import CSV data from the specified path.

    Args:
        pth (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Imported data as a pandas DataFrame.
    """
    return pd.read_csv(pth, index_col=0)


def perform_eda(df):
    """
    Perform exploratory data analysis on the DataFrame and save plots.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        None
    """
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plot_and_save(df['Churn'], 'Churn distribution', 'churn_distribution.png')
    plot_and_save(
        df['Customer_Age'],
        'Age distribution',
        'customer_age_distribution.png')
    plot_and_save(
        df.Marital_Status.value_counts('normalize'),
        'Marital status distribution',
        'marital_status_distribution.png',
        kind='bar')
    plot_and_save(
        df['Total_Trans_Ct'],
        'Total transaction distribution',
        'total_transaction_distribution.png',
        kind='hist')
    plot_correlation_heatmap(df)


def plot_and_save(data, title, filename, kind='hist'):
    """
    Create and save a plot.

    Args:
        data: Data to plot.
        title (str): Plot title.
        filename (str): Output filename.
        kind (str): Kind of plot ('hist' or 'bar').

    Returns:
        None
    """
    plt.figure(figsize=(20, 10))
    if kind == 'hist':
        plt.hist(data)
    elif kind == 'bar':
        data.plot(kind='bar')
    plt.title(title)
    plt.savefig(f'./images/eda/{filename}')
    plt.close()


def plot_correlation_heatmap(df):
    """
    Create and save a correlation heatmap for numeric columns.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        None
    """
    plt.figure(figsize=(20, 10))
    df_numeric = df.select_dtypes(include=['number'])
    sns.heatmap(df_numeric.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap.png')
    plt.close()


def encoder_helper(df, category_lst, response='Churn'):
    """
    Encode categorical columns with mean churn value.

    Args:
        df (pd.DataFrame): Input DataFrame.
        category_lst (list): List of categorical columns.
        response (str): Response variable name.

    Returns:
        pd.DataFrame: DataFrame with new encoded columns.
    """
    for category in category_lst:
        df[f'{category}_{response}'] = df.groupby(
            category)[response].transform('mean')
    return df


def perform_feature_engineering(df, response='Churn'):
    """
    Perform feature engineering on the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        response (str): Response variable name.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    df = encoder_helper(df, cat_columns, response)

    y = df[response]
    X = df[KEEP_COLS]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)


def classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf):
    """
    Produce classification reports for training and testing results and store plots.

    Args:
        y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf: Various prediction data.

    Returns:
        None
    """
    for name, train_preds, test_preds in [('Random Forest', y_train_preds_rf, y_test_preds_rf), (
            'Logistic Regression', y_train_preds_lr, y_test_preds_lr)]:
        plt.figure(figsize=(8, 8))
        plt.text(
            0.01, 1.1, f'{name} Train', {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    y_train, train_preds)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.6, f'{name} Test', {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.2, str(
                classification_report(
                    y_test, test_preds)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(
            f'./images/results/{name.lower().replace(" ", "_")}_results.png')
        plt.close()


def feature_importance_plot(model, X_data, output_pth):
    """
    Create and store feature importance plot.

    Args:
        model: Trained model.
        X_data (pd.DataFrame): Input features.
        output_pth (str): Output path for the plot.

    Returns:
        None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 16))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.close()


class Classifier:
    """Base class for classifiers."""

    def __init__(self):
        self.model = None
        self.name = ""

    def train_model(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)

    def save_model(self):
        """Save the model."""
        joblib.dump(self.model, f'./models/{self.name}_model.pkl')


class LRClassifier(Classifier):
    """Logistic Regression classifier."""

    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.name = 'logistic'


class RFClassifier(Classifier):
    """Random Forest classifier."""

    def __init__(self):
        super().__init__()
        self.estimator = RandomForestClassifier(random_state=42)
        self.name = 'rfc'
        self.param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

    def train_model(self, X_train, y_train):
        """Train the model using GridSearchCV."""
        cv_rfc = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv=5)
        cv_rfc.fit(X_train, y_train)
        self.model = cv_rfc.best_estimator_


def train_models(X_train, X_test, y_train, y_test):
    """
    Train models and produce performance plots.

    Args:
        X_train, X_test, y_train, y_test: Training and test data.

    Returns:
        None
    """
    rfc = RFClassifier()
    lrc = LRClassifier()

    rfc.train_model(X_train, y_train)
    lrc.train_model(X_train, y_train)

    plot_roc_curves(rfc.model, lrc.model, X_test, y_test)

    y_train_preds_rf = rfc.model.predict(X_train)
    y_test_preds_rf = rfc.model.predict(X_test)
    y_train_preds_lr = lrc.model.predict(X_train)
    y_test_preds_lr = lrc.model.predict(X_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
    feature_importance_plot(
        rfc.model,
        X_train,
        './images/results/feature_importances.png')

    rfc.save_model()
    lrc.save_model()


def plot_roc_curves(rfc_model, lr_model, X_test, y_test):
    """
    Plot ROC curves for Random Forest and Logistic Regression models.

    Args:
        rfc_model: Trained Random Forest model.
        lr_model: Trained Logistic Regression model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        None
    """
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    RocCurveDisplay.from_estimator(lr_model, X_test, y_test, ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')
    plt.close()


def get_models():
    """
    Load pre-trained models.

    Returns:
        tuple: Logistic Regression model, Random Forest model
    """
    lr_model = joblib.load('./models/logistic_model.pkl')
    rfc_model = joblib.load('./models/rfc_model.pkl')
    return lr_model, rfc_model


if __name__ == '__main__':
    df = import_data("./data/bank_data.csv")
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    train_models(X_train, X_test, y_train, y_test)
