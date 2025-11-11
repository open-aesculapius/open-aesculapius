# Примеры новых вариантов использования кода  [![en](https://img.shields.io/badge/en-ru-green.svg)](examples.md)

Примеры в этом разделе демонстрируют работу отдельных библиотечных функций

| Filename                                    | Description                                      |
|---------------------------------------------|--------------------------------------------------|
| simple_detect_perineural_cells.py         | `Детектирование ячейки на стеке размером 32x32`  |
| simple_find_perineural_cell_area.py       | `Нахождение границы ячейки`                      |
| simple_calculate_perineural_cell_depth.py | `Вычисление глубины найденной ячейки`            |
| simple_detect_astrocyte_tips.py             | `Детектирование кончиков отростков астроцитов`   |
| simple_classify_astrocyte.py                | `Классификация астроцитов на больные и здоровые` |
| simple_segment_gfap_microstructure.py       | `Сегментация микроструктуры GFAP цитоскелета`    |

| Filename                                          | Description                                                |
|---------------------------------------------------|------------------------------------------------------------|
| simple_denoise_ultrasound_image.py                | `Вычисление уровня УЗИ-шума(speckle) и его удаление`       |
| simple_detect_ultrasound_artifacts.py             | `Детектирование статических и динамических артефактов`     |
| simple_remove_ultrasound_artifacts.py             | `Удаление найденных статических и динамических артефактов` |
| simple_remove_mirror_ultrasound_artifact.py       | `Нахождение и удаление эффекта двоения`                    |
| simple_update_ultrasound_brightness_contrast.py   | `Автоматическое изменение уровней яркости/контрастности`   |
| simple_update_ultrasound_color_count_manual.py    | `Настраиваемое изменение колчества отображаемых цветов`    |
| simple_update_ultrasound_brightness_manual.py     | `Настраиваемое изменение яркости`                          |
| simple_update_ultrasound_contrast_manual.py       | `Настраиваемое изменение контрастности`                    |
| simple_apply_ultrasound_histogram_equalization.py | `Применение гистограммной эквализации`                     |
| simple_detect_ultrasound_anomalies.py             | `Детектирование аномалий`                                  |
| simple_segment_thyroid_image.py                   | `Детектирование краев(щитовидной железы)`                  |

Следующие примеры могут быть использованы для дальнейшего анализа найденных ячеек перинейроныльных сетей:

- simple_detect_perineural_cells.py;
- simple_find_perineural_cell_area.py;
- simple_calculate_perineural_cell_depth.py.

Следующие примеры могут быть использованы для дальнейшего анализа найденных астроцитов:

- simple_classify_astrocyte.py;
- simple_detect_astrocyte_tips.py;
- simple_segment_gfap_microstructure.py.

Следующие примеры могут быть использованы в качестве предобработки или постобработки в каскаде алгоритмов для работы с УЗИ-снимками:

- simple_apply_ultrasound_histogram_equalization.py;
- simple_denoise_ultrasound_image.py;
- simple_update_ultrasound_brightness_contrast.py;
- simple_update_ultrasound_brightness_manual.py;
- simple_update_ultrasound_color_count_manual.py;
- simple_update_ultrasound_ contrast_manual.py.

Следующие примеры могут быть использованы для автоматического выделения диагностически значимых областей:

- simple_detect_ultrasound_anomalies.py;
- simple_detect_ultrasound_artifacts.py;
- simple_remove_mirror_ultrasound_artifact.py;
- simple_remove_ultrasound_artifacts.py;
- simple_segment_thyroid_image.py.
