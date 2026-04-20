from catboost import CatBoostClassifier, Pool
from matplotlib import pyplot as plt
import json
import shap
import sklearn
from lightgbm import LGBMClassifier
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
from pygam import LogisticGAM
from sklearn.metrics import accuracy_score
from utils import DataProcessor
import pickle
from pygam import GAM, s, te

def create_and_fit_catboost(data):
    train_dataset = Pool(data.x_train,
                         label=data.y_train)

    model = CatBoostClassifier(depth=8, learning_rate=0.15, l2_leaf_reg=3, iterations=100,
                               early_stopping_rounds=30, verbose=False, eval_metric='Accuracy', thread_count=-1)
    model.fit(train_dataset)
    return model


def create_and_fit_lgb(data):
    lgb_model = LGBMClassifier(verbose=-1, colsample_bytree=0.55, depth=1, iterations=7, l2_leaf_reg=1,
                               learning_rate=0.415,
                               min_child_samples=180, num_leaves=10, reg_alpha=0, subsample=0.45)
    lgb_model.fit(data.x_train, data.y_train)
    return lgb_model


def create_and_fit_xgboost(data):
    model = xgb.XGBClassifier(objective='binary:logistic')
    model.fit(data.x_train, data.y_train)
    return model

def create_fit_gam(data):
    with open('Data/explainers/cat_top_features.json', 'r') as f:
        cat_top_features = json.load(f)

    top_features = cat_top_features

    data.x_train = data.x_train[top_features]
    data.x_test = data.x_test[top_features]

    terms = s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9)

    gam = LogisticGAM(terms).gridsearch(data.x_train.values, data.y_train)
    return gam

def use_grid_search_lgb(model):
    grid = {
        'colsample_bytree': [0.5495, 0.55],
        'depth': [1],  # depth=1 всегда лучший
        'iterations': [4, 7, 10],
        'l2_leaf_reg': [1, 2],
        'learning_rate': [0.416, 0.417],
        'min_child_samples': [175, 180],
        'num_leaves': [10, 12],
        'reg_alpha': [0, 0.001],
        'subsample': [0.452, 0.453, 0.455],
        'min_split_gain': [0.0, 0.05],
        'subsample_freq': [0, 1],
    }

    grid_search = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(data.x_train, data.y_train)

    print("Лучшие параметры:", grid_search.best_params_)
    print("Лучший результат:", grid_search.best_score_)

def use_grid_search_gam():

    with open('Data/explainers/top_features.json', 'r') as f:
        orig_top_features = json.load(f)

    for count in [7, 10, 13, 15, 17]:
        top_features = orig_top_features[:count]

        X_train = data.x_train[top_features]
        X_test = data.x_test[top_features]

        terms = s(0)
        for i in range(1, len(top_features)):
            terms += s(i)

        model = LogisticGAM(terms)

        gam = model.gridsearch(
            X_train.values,
            data.y_train,
            n_splines=[10, 25, 35, 45, 50],
            max_iter=[100, 300, 600, 2000],
            fit_intercept=[True, False]
        )

        y_pred = gam.predict(X_test.values)

        # Точность
        accuracy = accuracy_score(data.y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

    joblib.dump(gam, f'Models/gam.pkl')

    return 0

def use_grid_search_xgboost(model):
    grid = {
        'max_depth': [6],
        'min_child_weight': [5],
        'gamma': [0.2],
        'colsample_bytree': [0.8],
        'subsample': [0.8],
        'eta': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        'n_estimators': [300, 500, 700, 1000, 1200, 1500],
        'reg_alpha': [0, 0.01, 0.05, 0.1],
        'reg_lambda': [0.5, 1, 1.5, 2]
    }
    grid_search = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(data.x_train, data.y_train)

    print("Лучшие параметры:", grid_search.best_params_)
    print("Лучший счет:", grid_search.best_score_)

def use_grid_search_catbootst(model):
    train_dataset = Pool(data.x,
                         label=data.y,
                         cat_features=data.cat_features)

    grid = {
        'learning_rate': [0.05, 0.1, 0.15],
        'depth': [8, 10, 12],
        'l2_leaf_reg': [1, 3, 5, 9],
        'iterations': [100, 200, 300]}

    grid_search_result = model.grid_search(param_grid=grid, X=train_dataset, plot=False, cv=3)
    print("Лучшие гиперпараметры", grid_search_result['params'])

def using_shap(model, path):
    explainer = shap.TreeExplainer(model)
    explanation = explainer(data.x_test)
    with open(path + "/shap_explanation.pkl", 'wb') as f:
        pickle.dump(explanation, f)


def using_model(model_name, shap=True):
    base_path = f'Data/explainers/{model_name}'
    model = joblib.load(f'Models/{model_name.lower()}.pkl')
    data.metrics_model(model.predict(data.x_test), model.predict_proba(data.x_test), model_name)
    print(f"Модель - {model_name}")
    if os.path.isdir(base_path) and shap:
        using_shap(model, base_path)


if __name__ == '__main__':
    #использовать окружение - venv_11
    data = DataProcessor("Data/hyper_wheat_ds_ch_norm_prep_mode=dai.csv").split_data()

    #Catboost
    using_model("CatBoost")

    # XGBoost
    using_model("XGBoost")

    # Light
    using_model("LightGBM")

    # GAM
    create_fit_gam(data)