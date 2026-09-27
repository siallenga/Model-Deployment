# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# for model serialization
import joblib
# for creating a folder
import os
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()

# Define numeric and categorical features
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
    'MaritalStatus', 'Designation'
]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost Classifier
xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')

# Small grid so the pipeline runs fast on GitHub Actions.
# Widen this if you want a more thorough hyperparameter search.
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.05, 0.1],
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_roc_auc", mean_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Best model
    best_model = grid_search.best_estimator_
    print("Best params:", grid_search.best_params_)

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    y_proba_train = best_model.predict_proba(Xtrain)[:, 1]
    y_proba_test = best_model.predict_proba(Xtest)[:, 1]

    # Metrics
    train_accuracy = accuracy_score(ytrain, y_pred_train)
    test_accuracy = accuracy_score(ytest, y_pred_test)

    train_precision = precision_score(ytrain, y_pred_train)
    test_precision = precision_score(ytest, y_pred_test)

    train_recall = recall_score(ytrain, y_pred_train)
    test_recall = recall_score(ytest, y_pred_test)

    train_f1 = f1_score(ytrain, y_pred_train)
    test_f1 = f1_score(ytest, y_pred_test)

    train_roc_auc = roc_auc_score(ytrain, y_proba_train)
    test_roc_auc = roc_auc_score(ytest, y_proba_test)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "train_ROC_AUC": train_roc_auc,
        "test_ROC_AUC": test_roc_auc
    })

    # Save next to app.py so the Streamlit app can load it directly, and log
    # it as an MLflow artifact for traceability
    model_path = "tourism_project/deployment/best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
