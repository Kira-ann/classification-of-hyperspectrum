import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from matplotlib import pyplot as plt
import lime
import shap
import sklearn
from lightgbm import LGBMClassifier
import xgboost as xgb
import joblib
import os
from utils import DataProcessor
import pickle

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

    #   0.925476603119584
    #   0.9306759098786829      model = xgb.XGBClassifier(objective='binary:logistic')
    #   0.8994800693240901      model = xgb.XGBClassifier(objective='binary:hinge')
    #   0.9220103986135182      model = xgb.XGBClassifier(objective='binary:logitraw')
    model.fit(data.x_train, data.y_train)
    return model


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


def use_grid_search_xgboost(model):
    # Параметры для СЕТКИ
    #   0.8510000000000000 - 'colsample_bytree': 0.85, 'eta': 0.03, 'gamma': 0, 'max_depth': 7, 'min_child_weight': 3, 'subsample': 0.6
    #   0.8784291871772908 - 'colsample_bytree': 0.85, 'eta': 0.05, 'gamma': 0.1, 'max_depth': 6, 'min_child_weight': 3, 'subsample': 0.6
    #   0.8784291871772908 - 'colsample_bytree': 0.85, 'eta': 0.05, 'gamma': 0.1, 'max_depth': 6, 'min_child_weight': 3, 'subsample': 0.6
    #   0.8862968963741918 - 'colsample_bytree': 0.86, 'eta': 0.05, 'gamma': 0.1, 'max_depth': 6, 'min_child_weight': 4, 'subsample': 0.61
    #   0.8871712136900318 - 'colsample_bytree': 0.86, 'eta': 0.047, 'gamma': 0.1, 'max_depth': 5, 'min_child_weight': 4, 'subsample': 0.6
    #   0.8919768056515872 - 'colsample_bytree': 0.865, 'eta': 0.047, 'gamma': 0.1, 'max_depth': 5, 'min_child_weight': 4, 'subsample': 0.6
    #   0.8919768056515872 - 'colsample_bytree': 0.863, 'eta': 0.047, 'gamma': 0.1, 'max_depth': 5, 'min_child_weight': 4, 'subsample': 0.6}
    #   0.8919768056515872 - 'colsample_bytree': 0.863, 'eta': 0.047, 'gamma': 0.0995, 'max_depth': 5, 'min_child_weight': 4, 'subsample': 0.6

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
    # 'colsample_bytree': 0.85, 'eta': 0.03, 'gamma': 0, 'max_depth': 7, 'min_child_weight': 3, 'subsample': 0.6
    grid_search = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(data.x_train, data.y_train)

    print("Лучшие параметры:", grid_search.best_params_)
    print("Лучший счет:", grid_search.best_score_)


def use_grid_search_catbootst(model):
    # Лучшие гиперпараметры {'depth': 8, 'learning_rate': 0.15, 'l2_leaf_reg': 3, 'iterations': 100}
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


def save_explainer_shap(model):

    explainer = shap.TreeExplainer(model)
    explanation = explainer(data.x_test)

    with open('shap_explanation.pkl', 'wb') as f:
        pickle.dump(explanation, f)


def save_table_shap_values(path, shap_values):
    # тут глобальные признаки для каждого объекта
    shap_df = pd.DataFrame(shap_values.values, columns=data.x_test.columns)

    shap_df.to_csv(f'{path}/shap_values_all.csv', index=False)
    shap_df.to_excel(f'{path}/shap_values_all.xlsx', index=False)

    # Для глабольной храрктеристики используются среднее абсолютных значенеий шепли(модуль)
    shap_global = np.mean(np.abs(shap_values.values), axis=0)

    global_shap_df = pd.DataFrame({
        'feature': data.x_test.columns,
        'global_shap_importance': shap_global
    }).sort_values('global_shap_importance', ascending=False)

    global_shap_df.to_csv(f'{path}/global_shap.csv', index=False)


def using_shap(model, path):
    explainer = shap.TreeExplainer(model)
    explanation = explainer(data.x_test)
    with open('shap_explanation.pkl', 'wb') as f:
        pickle.dump(explanation, f)

def save_plots_lime(model, path, discretize_continuous=False):
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=data.x_train.values,
                                                       feature_names=data.x.columns,
                                                       class_names=['0', '1'],
                                                       discretize_continuous=discretize_continuous)
    for i in range(len(data.y_test)):
        exp = explainer.explain_instance(data.x_test.iloc[i].values, model.predict_proba, num_features=15)
        fig = exp.as_pyplot_figure(label=1)
        fig.set_size_inches(12, 8)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{path}/object_{i}.png')
        plt.close()

def using_model(model_name):
    base_path = f'Data/explainers/{model_name}'
    path_for_shap = os.path.join(base_path, 'SHAP')
    path_for_lime = os.path.join(base_path, 'LIME_exact')
    path_for_lime_exact = os.path.join(base_path, 'LIME')
    model = joblib.load(f'Models/{model_name.lower()}.pkl')
    data.metrics_model(model.predict(data.x_test), model.predict_proba(data.x_test))
    # if os.path.isdir(path_for_shap):
    #     using_shap(model, path_for_shap)
    # if os.path.isdir(path_for_lime):
    #     save_plots_lime(model, path_for_lime, discretize_continuous=True)
    # if os.path.isdir(path_for_lime_exact):
    #     save_plots_lime(model, path_for_lime_exact)
    # joblib.dump(model, f'Models/{model_name.lower()}.pkl')


if __name__ == '__main__':
    data = DataProcessor("Data/hyper_wheat_ds_ch_norm_prep_mode=dai.csv").split_data()

    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
