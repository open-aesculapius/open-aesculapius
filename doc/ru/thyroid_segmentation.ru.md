# Сегментация щитовидной железы на УЗ-снимках [![en](https://img.shields.io/badge/en-ru-green.svg)](../en/thyroid_segmentation.md)

Данный файл содержит описание метода автоматической сегментации области щитовидной железы на ультразвуковых изображениях.

---

## Основная идея
Алгоритм принимает серое ультразвуковое изображение, масштабирует его до 256 × 256, нормализует и передаёт в сверточную нейронную сеть **T‑Net** (модифицированный U‑Net с каналным вниманием).  
Сеть возвращает вероятностную карту; далее выполняются:

1. **Бинаризация** (порог 0.5)  
2. **Морфологическая очистка** (median‑blur → open → close)  
3. **Фильтрация по площади** (удаление областей < 500 px²)

На выходе — бинарная маска щитовидной железы и изображение с наложенным контуром/полупрозрачной маской.

---

## Демонстрация
Для демонстрации работы модуля используйте пример  
```
<корень_проекта>/examples/simple_segment_thyroid_image.py
```

**Входное изображение:**

![raw thyroid ultrasound](/doc/assets/raw_thyroid_segmentation_ultrasound.png)

**Полученная бинарная маска:**

![mask thyroid ultrasound](/doc/assets/result_thyroid_segmentation_mask.png)

**Наложение маски на оригинал:**

![overlay thyroid ultrasound](/doc/assets/result_thyroid_segmentation_overlay.png)