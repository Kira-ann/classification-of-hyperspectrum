from kan import KAN
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import DataProcessor
import numpy as np
from itertools import product
import torch, os
import shap
import pickle

def metrics_model(data, test_predictions, test_predictions_proba):
    accuracy = accuracy_score(data.y_test, test_predictions)
    print("Доля правильных ответов: ", accuracy)
    cnf_matrix = confusion_matrix(data.y_test, test_predictions)
    print(cnf_matrix)
    report = classification_report(data.y_test, test_predictions, target_names=['0', '1'], digits=4)
    print(report)


def grid_search(dataset):
    param_grid = {
        'width': [[318, 159, 80, 20, 6, 2]],
        'grid_val': [8],
        'k': [3],
        'steps': [50],
        'lr': [0.001]
    }

    configs = list(product(
        param_grid['width'],
        param_grid['grid_val'],
        param_grid['k'],
        param_grid['steps'],
        param_grid['lr']
    ))

    for idx, (width, grid_val, k, steps, lr) in enumerate(configs, 1):
        model = KAN(width=width, grid=grid_val, k=k)

        model.fit(
            dataset,
            opt="Adam",
            steps=steps,
            lr=lr,
            loss_fn=torch.nn.CrossEntropyLoss(),
        )


def using_shap(model, X_train, X_test, data, path):
    model.eval()
    background = shap.sample(X_train, 20, random_state=42)
    X_explain = shap.sample(X_test, 20, random_state=0)

    def predict_proba(x):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32)
            logits = model(xt)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    explainer = shap.KernelExplainer(predict_proba, background)

    shap_values = explainer.shap_values(X_explain, nsamples=50)

    explanation_obj = shap.Explanation(
        values=shap_values[1],
        base_values=float(np.array(explainer.expected_value).ravel()[1]),
        data=X_explain,
        feature_names=data.x_train.columns.tolist())

    with open(os.path.join(path, "shap_explanation.pkl"), 'wb') as f:
        pickle.dump(explanation_obj, f)

if __name__ == '__main__':
    #использовать окружение - venv_kan
    data = DataProcessor("Data/hyper_wheat_ds_ch_norm_prep_mode=dai.csv").split_data()

    y_train = data.y_train.ravel().astype(np.int64)
    X_train = data.x_train.to_numpy().astype(np.float32)
    y_test = data.y_test.ravel().astype(np.int64)
    X_test = data.x_test.to_numpy().astype(np.float32)

    dataset = {
        'train_input': torch.tensor(X_train, dtype=torch.float32),
        'train_label': torch.tensor(y_train, dtype=torch.long),
        'test_input': torch.tensor(X_test, dtype=torch.float32),
        'test_label': torch.tensor(y_test, dtype=torch.long)
    }

    # grid_search(dataset, y_test)

    model = KAN(width=[318, 159, 80, 20, 6, 2], grid=8, k=3)

    model.fit(
        dataset,
        opt="Adam",
        steps=50,
        lr=0.001,
        loss_fn=torch.nn.CrossEntropyLoss(),
    )

    torch.save(model.state_dict(), 'Models/kan_state.pth')

    model = KAN(width=[318, 159, 80, 20, 6, 2], grid=8, k=3)
    model.load_state_dict(torch.load('Models/kan_state.pth'))
    model.eval()

    y_pred_logits = model(torch.tensor(X_test, dtype=torch.float32))
    y_pred_proba = torch.softmax(y_pred_logits, dim=1).detach().numpy()
    y_pred = y_pred_proba.argmax(axis=1)

    # metrics_model(data, y_pred, y_pred_proba)
    using_shap(model, X_train, X_test, data, "Data/explainers/KAN")