# Examples of new code use cases  [![ru](https://img.shields.io/badge/ru-en-green.svg)](examples.ru.md) 

Examples in this chapter demonstrate work of a specific library function

| Filename                                    | Description                              |
|---------------------------------------------|------------------------------------------|
| simple_detect_perineural_cells.py         | `Detect one cell on 32x32 stack`         |
| simple_find_perineural_cell_area.py       | `Find cell area on detected cell`        |
| simple_calculate_perineural_cell_depth.py | `Calculate cell depth on detected cells` |
| simple_detect_astrocyte_tips.py             | `Detect astrocyte's tips`                |
| simple_classify_astrocyte.py                | `Classify healthy and sick astrocytes`   |
| simple_segment_gfap_microstructure.py       | `Segment astrocyte microstructure`       |

| Filename                                          | Description                                     |
|---------------------------------------------------|-------------------------------------------------|
| simple_denoise_ultrasound_image.py                | `Estimate noise level and remove speckle noise` |
| simple_detect_ultrasound_artifacts.py             | `Detect static and dynamic artifacts`           |
| simple_remove_ultrasound_artifacts.py             | `Remove detected static and dynamics artifact`  |
| simple_remove_mirror_ultrasound_artifact.py       | `Find and remove mirror artifact`               |
| simple_update_ultrasound_brightness_contrast.py   | `Auto change brightness and contrast levels`    |
| simple_update_ultrasound_color_count_manual.py    | `Manual change image color count`               |
| simple_update_ultrasound_brightness_manual.py     | `Manual change brightness`                      |
| simple_update_ultrasound_contrast_manual.py       | `Manual change contrast`                        |
| simple_apply_ultrasound_histogram_equalization.py | `Apply histogram equalization`                  |
| simple_detect_ultrasound_anomalies.py             | `Detect anomalies`                              |
| simple_segment_thyroid_image.py                   | `Detect edges(thyroid)`                         |


The following examples can be used to further analyze the detected perineural network cells:

- simple_detect_perineural_cells.py;
- simple_find_perineural_cell_area.py;
- simple_calculate_perineural_cell_depth.py.

The following examples can be used to further analyze the detected astrocytes:

- simple_classify_astrocyte.py;
- simple_detect_astrocyte_tips.py;
- simple_segment_gfap_microstructure.py.

The following examples can be used as preprocessing or postprocessing in a cascade of algorithms for working with ultrasound images:

- simple_apply_ultrasound_histogram_equalization.py;
- simple_denoise_ultrasound_image.py;
- simple_update_ultrasound_brightness_contrast.py;
- simple_update_ultrasound_brightness_manual.py;
- simple_update_ultrasound_color_count_manual.py;
- simple_update_ultrasound_ contrast_manual.py.

The following examples can be used to automatically highlight diagnostically significant areas:

- simple_detect_ultrasound_anomalies.py;
- simple_detect_ultrasound_artifacts.py;
- simple_remove_mirror_ultrasound_artifact.py;
- simple_remove_ultrasound_artifacts.py;
- simple_segment_thyroid_image.py.