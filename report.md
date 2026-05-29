# Report

## Track

Выбранный трек:

```text
A
```

## Что реализовано

- [ ] dataset.py
- [ ] processor.py
- [ ] model.py
- [ ] train.py
- [ ] benchmark.py

## Конфигурация

```text
config path: configs/track_a_cpu.yaml
seed: 42
device: cpu
dtype: float32
max_steps: 3
batch size: 1
```

## Результаты

```text
public tests: 14/14
train loss: -
benchmark accuracy: -
```

## Использованные ресурсы

```text
CPU/GPU: CPU
VRAM: 0 GB
время обучения: -
```

## Анализ ошибок

Приведите 3 ошибки модели (путь А, поэтому модель не обучалась):

1. Забыла выключить плиту
2. Написала после всех переменных _gpt
3. epsilont

## Комментарии

Самое сложное было разобраться с размерностями тензоров, в общем все как обычно. Из улучшений можно было бы выбрать путь посложнее и обучить модельку)


## Критерии оценивания

См. файл [`GRADING.md`](GRADING.md).
