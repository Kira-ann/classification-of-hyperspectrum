# Интерпретация SHAP

Приведем пример на одном объекте в качетве примера. У шкалы Ox E[f(X)] написано среднее предсказание для объекта. 

![Image alt](https://github.com/Kira-ann/classification-of-hyperspectrum/raw/main/Data/explainers/Catboost/SHAP/object_0.png)

{username} — ваш ник на ГитХабе;
{repository} — репозиторий где хранятся картинки;
{branch} — ветка репозитория;
{path} — путь к месту нахождения картинки.

При выделении 16 наиболее важных признаков методом SHAP получилось:
  
  5 признаков важны для каждой модели: src_490, src_494, src_498, std_494, std_498
  
  13 признаков у двух моделей: minmax_654, minmax_658, src_490, src_494, src_498, src_562, src_578, src_582, src_586, std_470, std_494, std_498, std_502
