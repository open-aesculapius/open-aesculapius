# Удаление эффекта двоения на УЗИ-снимке [![en](https://img.shields.io/badge/en-ru-green.svg)](../en/remove_mirror_ultrasound_artifact.md)
Данный файл содержит описание метода удаления эффекта двоения

# Основная идея
Алгоритм принимает на вход изображение с артефактом двоения и начинает с поиска кандидатов на удаление, используя детектор границ, свертки и пороговые фильтры. Затем с помощью нейронной сети извлекаются признаки из полученных изображений кандидатов. Если расстояние между этими признаками оказывается достаточно малым, считается, что эти два кандидата являются двойниками. В этом случае более яркий кандидат заполняется маской и дополнительно корректируется с помощью генеративно-состязательной нейронной сети, учитывающей содержимое изображения. Итогом работы алгоритма является изображение, очищенное от артефакта двоения.

# Демонстрация
Для демонстрации работы модуля был использован пример `simple_remove_mirror_ultrasound_artifact.py`, расположенный по следующему пути:
```
<корень_проекта>/examples/simple_remove_mirror_ultrasound_artifact.py
```
Ниже представлено исходное изображение, которое было использовано в качестве входных данных. В зеленых областях находятся оригинальная и двойственная области:

![raw mirror ultrasound](/doc/assets/raw_mirror_ultrasound.png)    

Результат работы модуля:

![result mirror ultrasound](/doc/assets/result_mirror_ultrasound.png)