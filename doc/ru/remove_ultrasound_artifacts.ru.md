# Удаление артфеактов УЗИ-снимка [![en](https://img.shields.io/badge/en-ru-green.svg)](../en/remove_ultrasound_artifacts.md)
Данный файл содержит описание метода удаления артефактов УЗИ-снимка

# Основная идея
Алгоритм принимает на вход изображение и маску, затем с использованием генеративно-состязательной нейронной сети он заполняет область, указанную маской, с учётом исходного содержимого изображения. В результате работы алгоритма получается финальное изображение с заполненными областями.

# Демонстрация
Для демонстрации работы модуля был использован пример `simple_remove_ultrasound_artifacts.py`, расположенный по следующему пути:
```
<корень_проекта>/examples/simple_remove_ultrasound_artifacts.py
```
Ниже представлено исходное изображение, которое было использовано в качестве входных данных:

![raw remove artifact ultrasound](/doc/assets/raw_remove_artifact_ultrasound.png)    

Ниже представлена маска артефакта, которая была использована в качестве входных данных:

![raw remove artifact ultrasound](/doc/assets/raw_mask_remove_artifact_ultrasound.png)   

Результат работы модуля:
 
![result remove artifact ultrasound](/doc/assets/result_remove_artifact_ultrasound.png)