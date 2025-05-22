# Aesculapius [![en](https://img.shields.io/badge/en-ru-green.svg)](README.md)

Aesculapius - открытая библиотека для обработки микроскопических и УЗИ-снимков.

# Содержание
- [Установка](#installation)
    - [Установка с github](#installation_from_github)
    - [Установка и сборка пакета](#installation_from_package)
- [Конфигурация git hooks](#git-hooks)
    - [MacOS или Linux](#hooks-unix)
    - [Windows](#hooks-windows)
- [Использование функций библиотеки](#how-to-use)
- [Примеры](#examples)
- [Лицензия](#license)
# <a name="installation">📝 Установка</a>
## <a name="installation_from_github">Установка с github</a>
1) Скачать и установить Python 3.10+
2) Скачать и установить Cuda 12.0+
3) Скачивание репозитория
```shellscript
git clone git@github.com:egorchevanton/open_aesculapius_private.git
```
4) Создание виртуального окружения
```shellscript
pip install virtualenv
virtualenv venv3 -p python3
```
5) Активация виртуального окружения

Windows:
```shellscript
venv3\Scripts\activate.bat
```
Linux:
```shellscript
source venv3/bin/activate
```
6) Установка зависимостей
```shellscript
pip install -r requirements.txt
```
7) Скачивание весов
```shellsctopy
python download_weights.py
```
## <a name="installation_from_package">Установка и сборка пакета</a>
1) Скачать и установить Python 3.10+
2) Скачать и установить Cuda 12.0+
3) Собрать пакет whl. В папке build:
- Для Windows: 
```shellscript
build.bat
```
- Для Ubuntu: 
```shellscript
./build.sh
```
4) Установка пакета
```shellscript
pip install open_aesculapius.whl
```
# <a name="git-hooks">📝 Конфигурация git hooks</a>
## Для проверки сообщений коммита на соответствие шаблону '\<type>[scope]: \<description>' были добавлены git hooks. В этом разделе описаны этапы конфигурации для их использования.
## <a name="hooks-unix">MacOS или Linux</a>
1. Сделать файл .githooks/commit-msg исполняемым
```shellscript
chmod +X .githooks/commit-msg
```
2. Запустить файл .githooks/hooks-init.sh
```shellscript
. .githooks/commit-msg
```
## <a name="hooks-windows">Windows</a>
1. Запустить файл через PowerShell .githooks/hooks-init.sh
```shellscript
. .githooks/commit-msg
```
# <a name="how-to-use">📝 Использование функций библиотеки</a>
Библиотека представляет два основных класса для работы с изображениями: 
- MicroscopicImage - обработка микроскопических изображений 
- UltrasoundImage - обработка УЗИ изображений 

```shellscript
from aesculapius.modules.core.ms_img import MicroscopicImage
from aesculapius.modules.core.us_img import UltrasoundImage

m_image = MicroscopicImage("путь_к_папке_с_изображениями")
layers = m_image.get_layers()

us_image = UltrasoundImage("путь_к_папке_с_изображениями/или_к_изображению")
us_layer = us_image.us_image
```
Пример вызова функций из библиотеки: 

```shellscript
from aesculapius.aesculapius import (
    detect_perineuronal_cells,
    denoise_ultrasound_image,
    segment_gfap_microstructure,
)

cells = detect_perineuronal_cells(layers, config)

denoised = denoise_ultrasound_image(us_layer, config)

segmented = segment_gfap_microstructure(layers, config)
```

Функции используют YAML-конфиги для настройки параметров.
Пример конфига (calculate_perineuronal_cell_depth.yaml):

```shellscript
folder_path: ../data/microscopic/perineuronal/cells_stack
out_path: ../data/out/calculate_perineuronal_cell_depth
weights: ../weights/cell_model.pkl
threshold: 0.7
```

# <a name="examples">📝 Примеры</a>
В директории [examples](examples) находятся примеры, демострирующие функционал библиотеки Aesculapius. Полный список примеров представлен в [examples.ru.md](examples/examples.ru.md) файле.
# <a name="license">📝 Лицензия</a>
Этот проект распространяется под GPL-3.0 лицензией