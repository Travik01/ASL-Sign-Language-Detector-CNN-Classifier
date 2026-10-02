# ASL Sign Language Detector — CNN Classifier

A convolutional neural network for detecting American Sign Language (ASL) gestures from 64×64 RGB images. The model classifies 35 hand sign classes (A–Z, space, del, nothing) using a two-layer CNN with dropout regularization. Includes a complete training pipeline with evaluation, confusion matrix visualization, and automatic model export.

---

## Features

- **35-class classification** — full ASL alphabet plus special tokens (space, del, nothing)
- **CNN architecture** — Conv2D 32 → Conv2D 64 → Dense 128 → Dense 35 (softmax)
- **Data preprocessing** — automatic image loading, resizing to 64×64, normalization to 0–1
- **Early stopping** — prevents overfitting with patience = 5
- **Evaluation** — classification report, confusion matrix, training history plots
- **Model export** — saves trained model as `.keras`

---

## Dataset

Download the dataset from Kaggle: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

Unzip into a folder named `Новая папка (11)` next to the script.

Expected structure:
```
Новая папка (11)/
├── A/
├── B/
├── C/
│   ...
├── Y/
├── Z/
├── del/
├── nothing/
└── space/
```

Each subfolder contains `.jpg` images of the corresponding hand sign (200×200, resized to 64×64 during loading).

---

## Installation

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```text
tensorflow==2.16.1
scikit-learn==1.5.1
numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.1
seaborn==0.13.2
Pillow==10.4.0
```

---

## Usage

```bash
python main.py
```

The script will:

1. Load images from `Новая папка (11)/` (64×64, normalized to 0–1)
2. Encode labels and split 80/20 into train/test
3. Train a CNN for up to 20 epochs (early stopping enabled)
4. Save the model to `sign_language_model.keras`
5. Generate `training_history.png` and `confusion_matrix.png`

---

## Model Architecture

```
Conv2D(32, 3×3, relu) → MaxPool(2×2)
Conv2D(64, 3×3, relu) → MaxPool(2×2)
Flatten → Dense(128, relu) → Dropout(0.5)
Dense(35, softmax)
```

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr = 1e-3) |
| Loss | categorical_crossentropy |
| Batch size | 128 |
| Epochs | 20 (early stopping, patience = 5) |
| Image size | 64×64×3 |
| Classes | 35 |

---

## Output Files

| File | Description |
|------|-------------|
| `sign_language_model.keras` | Trained CNN model |
| `training_history.png` | Accuracy and loss curves |
| `confusion_matrix.png` | Confusion matrix heatmap (35×35) |

---

## Project Structure

```
asl-detector/
├── main.py                      # All code (data loading, training, evaluation)
├── requirements.txt             # Dependencies
├── Новая папка (11)/            # Dataset folder (unzip ASL Alphabet here)
├── sign_language_model.keras    # Trained model (generated after run)
├── training_history.png         # Training plots (generated after run)
├── confusion_matrix.png         # Confusion matrix (generated after run)
└── README.md
```

---

## Requirements

Python ≥ 3.9, TensorFlow ≥ 2.16, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn, Pillow

## License

MIT

---

# ASL Sign Language Detector — CNN-классификатор

Свёрточная нейрон сеть для распознавания жестов американского языка жестов (ASL) по изображениям 64×64. Модель классифицирует 35 классов жестов (A–Z, space, del, nothing) с помощью двухслойной CNN с dropout-регуляризацией. Включает пайплайн обучения, оценку метрик, визуализацию матрицы ошибок и автоматический экспорт модели.

---

## Возможности

- **Классификация 35 классов** — полный алфавит ASL плюс спецсимволы (space, del, nothing)
- **Архитектура CNN** — Conv2D 32 → Conv2D 64 → Dense 128 → Dense 35 (softmax)
- **Предобработка** — автоматическая загрузка, ресайз до 64×64, нормализация 0–1
- **Ранняя остановка** — защита от переобучения (patience = 5)
- **Оценка** — classification report, матрица ошибок, графики обучения
- **Экспорт модели** — сохранение в формате `.keras`

---

## Датасет

Скачайте датасет с Kaggle: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

Распакуйте в папку `Новая папка (11)` рядом со скриптом.

Ожидаемая структура:
```
Новая папка (11)/
├── A/
├── B/
├── C/
│   ...
├── Y/
├── Z/
├── del/
├── nothing/
└── space/
```

Каждая подпапка содержит `.jpg`-изображения соответствующего жеста (200×200, при загрузке ресайзятся до 64×64).

---

## Установка

```bash
pip install -r requirements.txt
```

Содержимое `requirements.txt`:
```text
tensorflow==2.16.1
scikit-learn==1.5.1
numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.1
seaborn==0.13.2
Pillow==10.4.0
```

---

## Использование

```bash
python main.py
```

Скрипт:

1. Загружает изображения из `Новая папка (11)/` (64×64, нормализация 0–1)
2. Кодирует метки и делит 80/20 на train/test
3. Обучает CNN до 20 эпох (с ранней остановкой)
4. Сохраняет модель в `sign_language_model.keras`
5. Генерирует `training_history.png` и `confusion_matrix.png`

---

## Архитектура модели

```
Conv2D(32, 3×3, relu) → MaxPool(2×2)
Conv2D(64, 3×3, relu) → MaxPool(2×2)
Flatten → Dense(128, relu) → Dropout(0.5)
Dense(35, softmax)
```

| Параметр | Значение |
|-----------|----------|
| Оптимизатор | Adam (lr = 1e-3) |
| Функция потерь | categorical_crossentropy |
| Размер батча | 128 |
| Эпохи | 20 (ранняя остановка, patience = 5) |
| Размер изображения | 64×64×3 |
| Классы | 35 |

---

## Выходные файлы

| Файл | Описание |
|------|----------|
| `sign_language_model.keras` | Обученная модель CNN |
| `training_history.png` | Графики accuracy и loss |
| `confusion_matrix.png` | Тепловая карта матрицы ошибок (35×35) |

---

## Структура проекта

```
asl-detector/
├── main.py                      # Весь код (загрузка, обучение, оценка)
├── requirements.txt             # Зависимости
├── Новая папка (11)/            # Папка с датасетом (распаковать ASL Alphabet)
├── sign_language_model.keras    # Обученная модель (создаётся после запуска)
├── training_history.png         # Графики обучения (создаётся после запуска)
├── confusion_matrix.png         # Матрица ошибок (создаётся после запуска)
└── README.md
```

---

## Зависимости

Python ≥ 3.9, TensorFlow ≥ 2.16, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn, Pillow

## Лицензия

MIT
