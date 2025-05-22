# Классификация астроцитов на больные и здоровые на конфокальных микроскопических снимках [![en](https://img.shields.io/badge/en-ru-green.svg)](../en/classify_astrocytes.md)

Данный файл содержит описание модуля классификации ранее обнаруженных астроцитов как здоровых или больных на основе конфокальных микроскопических изображений.

# Основная идея
Модуль принимает на вход маску изображения с выделенной областью астроцита и извлекает его морфологические признаки. Для бинарной классификации используется дообученная на ImageNet сверточная нейросеть VGG19, адаптированная под две категории и дополнительно обученная на аугментированном наборе данных здоровых и больных астроцитов. Для учёта дисбаланса классов применяется взвешенная функция потерь, а финальное решение принимается по выводу softmax и выбору argmax.

# Демонстрация
Для демонстрации работы модуля используется пример `simple_classify_astrocyte.py`, расположенный по следующему пути:
```
<корень_проекта>/examples/simple_classify_astrocyte.py
```

Ниже приведено исходное конфокальное микроскопическое изображение астроцита:

![raw classify_astrocytes](/doc/assets/raw_classify_astrocytes.png)

Результат работы модуля:

Модуль возвращает метку и вероятность принадлежности:
```
layer: 0: predicted class - healthy, probability - 0.7299
layer: 1: predicted class - healthy, probability - 0.8539
layer: 2: predicted class - healthy, probability - 0.5224
layer: 3: predicted class - healthy, probability - 0.8440
layer: 4: predicted class - healthy, probability - 0.9012
layer: 5: predicted class - healthy, probability - 0.7491
layer: 6: predicted class - healthy, probability - 0.9363
layer: 7: predicted class - healthy, probability - 0.9133
layer: 8: predicted class - healthy, probability - 0.9193
layer: 9: predicted class - healthy, probability - 0.7508
layer: 10: predicted class - healthy, probability - 0.8715
```