Программа для встраивания и извлечения скрытых водяных знаков
через FFT и Wavelet преобразования.

## Установка

    pip install -r requirements.txt

## Запуск GUI

    python main.py

## Статистический анализ

    python run_analysis.py --host фото.png --wm знак.png

## Структура проекта

    files/
    ├── main.py              # запуск GUI
    ├── run_analysis.py      # статистические тесты
    ├── core/                # алгоритмы FFT и Wavelet
    └── gui/                 # интерфейс PyQt6
