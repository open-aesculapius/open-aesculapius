# Aesculapius [![en](https://img.shields.io/badge/en-ru-green.svg)](README.md)

Aesculapius - открытая библиотека для обработки микроскопических и УЗИ-снимков.

# Содержание
- [Установка](#installation)
- [Конфигурация git hooks](#git-hooks)
    - [MacOS или Linux](#hooks-unix)
    - [Windows](#hooks-windows)
- [Использование функций библиотеки](#how-to-use)
- [Примеры новых вариантов использования кода](#examples)
- [Перечень направлений прикладного использования](#list)
- [Минимальные системные требования](#requirements)
- [Лицензия](#license)
# <a name="installation">📝 Установка</a>
1) Скачать и установить Python 3.10
2) Скачать и установить Cuda 12.0+
3) Скачивание репозитория
```shellscript
git clone git@github.com:open-aesculapius/open-aesculapius.git

cd open-aesculapius
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

По умолчанию для Windows устанавливается версия PyTorch для CPU. Для работы модулей 
на GPU посетите следующий [сайт](https://pytorch.org/get-started/previous-versions/) и установите нужную версию.

7) Сборка и установка пакета whl (опционально)

```shellscript
cd build 
```

- Для Windows: 
```shellscript
build.bat
```
- Для Ubuntu: 
```shellscript
./build.sh
```

- Установка собранного пакета (пакет появится в директории dist)
```shellscript
pip install aesculapius-1.0.0-py3-none-any.whl
```

8) Для проверки работоспособности всех модулей из директории test:
```shellscript 
cd test 
python module_testing.py
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

# <a name="examples">📝 Примеры новых вариантов использования кода</a>
В директории [examples](examples) находятся примеры, демострирующие функционал библиотеки Aesculapius. Полный список примеров представлен в [examples.ru.md](examples/examples.ru.md) файле.

Запуск примеров осуществляется с помощью следующей команды (из корневого каталога): 

```shellscript
python -m examples.<имя_модуля>
```
Пример: 
```shellscript
python -m examples.simple_apply_ultrasound_histogram_equalization 
```

# <a name="list">📝 Перечень направлений прикладного использования: </a>

Библиотека может быть применена в следующих прикладных направлениях:

• Медицинская визуализация — для улучшения качества УЗИ-снимков, включая повышение контрастности и подавление шумов, а также для автоматического выделения диагностически значимых областей.

• Анализ микроскопических изображений — библиотека включает модули для обработки и анализа микроскопических изображений, содержащих астроциты или ячейки перинейрональной сети.

• Научные и образовательные задачи — для воспроизведения результатов научных исследований, а также в качестве инструмента обучения специалистов в области цифровой медицины и биоинформатики.


# <a name="requirements">📝 Минимальные системные требования: </a>

• процессор Intel или AMD 2.3 ГГц x10 ядер; оперативная память: 8 Гб; видеокарта Nvidia RTX 4060; постоянная память SSD 512 Гб; версия ОС персонального компьютера Windows 10 или Ubuntu 22.04;

• наличие интерпретатора Python версии 3.10;

• установленные библиотеки из файла requirements.txt.


# <a name="license">📝 Лицензия</a>
Этот проект распространяется под GPL-3.0 лицензией