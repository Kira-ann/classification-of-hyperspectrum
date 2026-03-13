from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import re

class DataProcessor:
    def __init__(self, path):
        self.dataset = pd.read_csv(path).drop(columns=['ds_name', 'dai', 'box_i', 'class_generalized', 'file_path', 'sort'])
        self.dataset = self.dataset.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        self.x = self.y = None
        self.x_train = self.x_test = self.y_train = self.y_test = None

    def split_data(self):
        self.x = self.dataset.drop(columns=['class'])
        self.y = self.dataset["class"]  # по какому столбцу идет классификация
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)  # Преобразует 'c' в 0, 'е' в 1(для XGBoost, gam)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y,
            test_size=0.2,
            random_state=42)  # для адекватног результата сравнения, при каждом запуске одинаковое деление выборок

        return self

    def output_plot(self, current_number, class_obj):
        wavelength = [i for i in range(450, 874, 4)]

        start_src_mean = self.dataset.columns.get_loc('leafs_src_mean_450')
        end_src_mean = self.dataset.columns.get_loc('leafs_src_mean_870')

        start_minmax_mean = self.dataset.columns.get_loc('leafs_minmax_mean_450')
        end_minmax_mean = self.dataset.columns.get_loc('leafs_minmax_mean_870')

        start_std_mean = self.dataset.columns.get_loc('leafs_std_mean_450')
        end_std_mean = self.dataset.columns.get_loc('leafs_std_mean_870')

        leafs_src_mean = self.dataset.iloc[current_number, [i for i in range(start_src_mean, end_src_mean + 1)]].values
        leafs_minmax_mean = self.dataset.iloc[
            current_number, [i for i in range(start_minmax_mean, end_minmax_mean + 1)]].values
        leafs_std_mean = self.dataset.iloc[current_number, [i for i in range(start_std_mean, end_std_mean + 1)]].values

        sns.set_theme(style='whitegrid')
        plt.plot(wavelength, leafs_src_mean, label='leafs_src_mean')
        plt.plot(wavelength, leafs_minmax_mean, label='leafs_minmax_mean')
        plt.plot(wavelength, leafs_std_mean, label='leafs_std_mean')
        plt.xlim(450, 870)
        plt.title(f"Гиперспектральная кривая для объекта №{current_number}")
        plt.xlabel("Длины волн")
        plt.ylabel("Коэффициент отражения")
        plt.legend()
        if class_obj == 'e':
            plt.savefig(f'Data/pictures_e/{current_number}.png')

        if class_obj == 'c':
            plt.savefig(f'Data/pictures_c/{current_number}.png')

        plt.close()
        # plt.show(

    def build_all_plot(self):
        for i in range(len(self.dataset)):
            current_class = self.dataset['class'].iloc[i]
            self.output_plot(i, current_class)

    @staticmethod
    def plot_confusion_matrix(cm, classes):
        plt.imshow(cm, interpolation='nearest')
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig("Data/metrics/conf_matrix.png")
        plt.close()

    def plot_roc_curve(self, test_predictions_proba):
        plt.figure(figsize=(10, 8))
        fpr, tpr, thresholds = roc_curve(self.y_test, test_predictions_proba[:, 1], pos_label=1)
        lw = 4
        plt.plot(fpr, tpr, lw=lw, label='ROC curve ')
        plt.plot([0, 1], [0, 1])
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.savefig("ROC.png")
        plt.close()

    def metrics_model(self, test_predictions, test_predictions_proba):
        accuracy = accuracy_score(self.y_test, test_predictions)
        print("Доля правильных ответов: ", accuracy)

        cnf_matrix = confusion_matrix(self.y_test, test_predictions)
        self.plot_confusion_matrix(cnf_matrix, classes=['0', '1'])

        report = classification_report(self.y_test, test_predictions, target_names=['0', '1'],
                                       digits=4)  # digits - кол-во знаков после ,
        print(report)

        self.plot_roc_curve(test_predictions_proba)