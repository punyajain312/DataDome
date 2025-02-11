from flask import session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, explained_variance_score, mean_absolute_error, mean_squared_error, r2_score,accuracy_score

from pre_processing.main import main
from pre_processing.modules.transformation import transform
from utils import global_store

models = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(),
    "random_forest": RandomForestRegressor(),
    "svm": SVR(),
    "knn": KNeighborsRegressor(),
    "logistic_regression": LogisticRegression(),
    "gradient_boosting": GradientBoostingRegressor(),
    # "xgboost": XGBRegressor()
}

def pre_process_data(file_path):
    checkbox = global_store.global_data["checkbox"]
    df, _ = main(file_path, gen_syn_data=checkbox)
    return df

def train_predict_regression(data_csv, model_name, target):
    df = pre_process_data(data_csv)
    df, scaler = transform(df, target_column=target, task="prediction")

    df.to_csv(r"output\after_preprocess.csv")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = models.get(model_name)
    # ravel to convert to 1D array if needed
    model.fit(X_train, y_train.ravel())

    y_test_pred = model.predict(X_test)
    # **Inverse transform predicted and actual y values**
    y_test = scaler.inverse_transform(
        y_test.to_numpy().reshape(-1, 1)).flatten()
    y_test_pred = scaler.inverse_transform(
        np.array(y_test_pred).reshape(-1, 1)).flatten()

    # Compute metrics
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    explained_variance = explained_variance_score(y_test, y_test_pred)
    mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * \
        100  # Mean Absolute Percentage Error

    # Store results
    results = {
        "Model Coefficients": model.coef_.tolist() if hasattr(model, "coef_") else "Not applicable",
        "Intercept": model.intercept_.tolist() if hasattr(model, "intercept_") else "Not applicable",
        "Expected": y_test.tolist(),
        "Preds": y_test_pred.tolist(),
        "Performance Metrics": {
            "Mean Squared Error": mse,
            "Root Mean Squared Error": rmse,
            "Mean Absolute Error": mae,
            "Mean Absolute Percentage Error": mape,
            "R-squared Score": r2,
            "Explained Variance Score": explained_variance
        }
    }

    return results, y_test, y_test_pred


def train_predict_classification(data_csv, model_name, target):
    df = pre_process_data(data_csv)
    df, scaler = transform(df, target_column=target, task="classification")

    df.to_csv(r"output\clean_user_data_test.csv")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = models.get(model_name)
    # ravel to convert to 1D array if needed
    model.fit(X_train, y_train.ravel())

    y_test_pred = model.predict(X_test)

    y_test = y_test.astype(int)
    y_test_pred = y_test_pred.astype(int)
    # **Inverse transform predicted and actual y values**
    y_test = scaler.inverse_transform(y_test.to_numpy())
    y_test_pred = scaler.inverse_transform(np.array(y_test_pred))

    accuracy = accuracy_score(y_test, y_test_pred)
    class_report = classification_report(y_test, y_test_pred, output_dict=True)

    metrics = {
        "Accuracy": accuracy,
        "Classification Report": class_report
    }

    num_classes = len(np.unique(y_train))

    results = {
        "Model Type": "Binary Classification" if num_classes == 2 else "Multiclass Classification",
        "Feature Importances": model.coef_.tolist() if hasattr(model, "coef_") else "Not applicable",
        "Predictions on Test Data": y_test_pred.tolist(),
        "Performance Metrics": metrics
    }

    return results, y_test.tolist(), y_test_pred.tolist()


def regression_standard(data_csv, model_name, target):
    data = pd.read_csv(data_csv)
    X = data.drop(columns=[target])
    y = data[target]

    X = X.fillna(X.mean())

    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    model = models.get(model_name)
    # ravel to convert to 1D array if needed
    model.fit(X_train, y_train.ravel())

    y_test_pred = model.predict(X_test)

    # **Inverse transform predicted and actual y values**
    y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred = y_scaler.inverse_transform(
        y_test_pred.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    explained_variance = explained_variance_score(y_test, y_test_pred)
    mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * \
        100  # Mean Absolute Percentage Error

    # Store results
    results = {
        "Model Coefficients": model.coef_.tolist() if hasattr(model, "coef_") else "Not applicable",
        "Intercept": model.intercept_.tolist() if hasattr(model, "intercept_") else "Not applicable",
        "Expected": y_test.tolist(),
        "Preds": y_test_pred.tolist(),
        "Performance Metrics": {
            "Mean Squared Error": mse,
            "Root Mean Squared Error": rmse,
            "Mean Absolute Error": mae,
            "Mean Absolute Percentage Error": mape,
            "R-squared Score": r2,
            "Explained Variance Score": explained_variance
        }
    }

    return results, y_test, y_test_pred


def classification_standard(data_csv, model_name, target):
    # Load data
    data = pd.read_csv(data_csv)
    X = data.drop(columns=[target])
    y = data.iloc[target]

    X = X.fillna(X.mean())

    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)

    y_scaler = LabelEncoder()
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    model = models.get(model_name)
    # ravel to convert to 1D array if needed
    model.fit(X_train, y_train.ravel())

    y_test_pred = model.predict(X_test)
   
    accuracy = accuracy_score(y_test, y_test_pred)
    class_report = classification_report(y_test, y_test_pred, output_dict=True)

    metrics = {
        "Accuracy": accuracy,
        "Classification Report": class_report
    }

    # Calculate the number of unique classes
    num_classes = len(np.unique(y_train))

    print("CLASSIFICATIOn")

    results = {
        "Model Type": "Binary Classification" if num_classes == 2 else "Multiclass Classification",
        "Feature Importances": model.coef_.tolist() if hasattr(model, "coef_") else "Not applicable",
        "Predictions on Test Data": y_test_pred.tolist(),
        "Performance Metrics": metrics
    }

    return results, y_test, y_test_pred


def visualize_results(un_results, un_y_test, un_y_pred, pros_results, pros_test, pros_pred, save_path):
    fig, axes = plt.subplots(4, 2, figsize=(20, 20))
    axes = axes.flatten()

    for i, (yt, yp, title) in enumerate([
        (un_results["Expected"], un_results["Preds"], "Conventional Results"),
        (pros_results["Expected"], pros_results["Preds"],
         "Our Model processing Results")
    ]):
        # Convert lists to numpy arrays for element-wise subtraction
        yt = np.array(yt).flatten()
        yp = np.array(yp).flatten()
        residuals = yt - yp

        sns.scatterplot(x=yt, y=yp, alpha=0.7, ax=axes[i])
        axes[i].plot([yt.min(), yt.max()], [
                     yt.min(), yt.max()], 'r', linestyle='--')
        axes[i].set_xlabel("Actual Values")
        axes[i].set_ylabel("Predicted Values")
        axes[i].set_title(f"{title} - Actual vs Predicted Values")

        sns.histplot(residuals, kde=True, bins=30,
                     color='blue', alpha=0.7, ax=axes[i + 2])
        axes[i + 2].set_xlabel("Residuals")
        axes[i + 2].set_title(f"{title} - Residuals Distribution")

        sns.boxplot(y=residuals, ax=axes[i + 4])
        axes[i + 4].set_title(f"{title} - Boxplot of Residuals")

        sns.lineplot(x=range(len(yt)), y=yt, label='Actual',
                     marker='o', ax=axes[i + 6])
        sns.lineplot(x=range(len(yp)), y=yp, label='Predicted',
                     marker='s', ax=axes[i + 6])
        axes[i + 6].set_xlabel("Sample Index")
        axes[i + 6].set_ylabel("Values")
        axes[i + 6].set_title(f"{title} - Actual vs Predicted Line Plot")
        axes[i + 6].legend()

    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

    return save_path
