# Удаление шумов с УЗИ-снимка [![en](https://img.shields.io/badge/en-ru-green.svg)](../en/denoise_ultrasound_image.md)
Данный файл содержит описание метода удаления speckle шумов

# Основная идея
Алгоритм удаления шумов принимает изображение и пропускает его через нейронную сеть типа автоэнкодер. Эта нейронная сеть была предварительно обучена на наборе данных, содержащем искусственно зашумленные УЗИ-снимки. Результатом работы алгоритма является изображение, очищенное от шумов.

# Демонстрация
Для демонстрации работы модуля был использован пример `simple_denoise_ultrasound_image.py`, расположенный по следующему пути:

```
<корень_проекта>/examples/simple_denoise_ultrasound_image.py
```

Ниже представлено исходное изображение, которое было использовано в качестве входных данных:

![raw denoise ultrasound](/doc/assets/raw_denoise_ultrasound.png)    

Результат работы модуля - УЗИ-изображение с уменьшенным количеством шума:

![result denoise  ultrasound](/doc/assets/result_denoise_ultrasound.png)

Вывод в консоль определения уровня шума:

![result denoise level ultrasound](/doc/assets/result_denoise_level_ultrasound.png)   

