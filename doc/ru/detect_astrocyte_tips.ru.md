# Распознавания кончиков отростков астроцитов на конфокальном микроскопическом снимке [![en](https://img.shields.io/badge/en-ru-green.svg)](../en/detect_astrocyte_tips.md)
Данный файл содержит описание модуля распознавания кончиков отростков астроцитов на конфокальных микроскопических снимках.

# Основная идея
Модуль распознавания кончиков отростков астроцитов на конфокальном микроскопическом снимке разработан для автоматического поиска кончиков отростков астроцитов на каждом из слоев общего микроскопического снимка астроцита. Алгоритм использует комбинацию классических методов компьютерного зрения и методов глубокого обучения для точного выделения и детектирования терминальных точек отростков. Использование гибридного подхода позволяет компенсировать недостатки отдельных методов и достичь высокой точности даже в случае сложных переплетений отростков.

# Демонстрация
Для демонстрации работы модуля был использован пример `simple_detect_astrocyte_tips.py`, расположенный по следующему пути:

```
<корень_проекта>/examples/simple_detect_astrocyte_tips.py
```
Ниже представлено исходное изображение, которое было использовано в качестве входных данных:

![raw astrocyte tips](/doc/assets/raw_detect_astrocyte_tips.png)

Результат работы модуля - изображение с распознанными кончиками отростков астроцитов:

![result detect astrocyte tips](/doc/assets/result_detect_astrocyte_tips.png)
