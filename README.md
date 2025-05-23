# Low-rank self-play fine-tuning for small LLMs

----

__Expert:__ Andrey Grabovoy

__Consultant:__ Nikita Okhotnikov

----

## Abstract

В работе исследуется проблема дообучения больших языковых моделей (LLM) в условиях ограниченных ресурсов. Под ограниченными ресурсами понимается видеопамять, человеческое участие и время обучения. В работе рассматриваются модели до 0.5B. Предлагается метод дообучения, основанный на внедрении адаптеров LoRA, малоранговых разложений матриц, в слои архитектуры трансформера, и использовании стратегии self-play - текущая итерация генерирует предсказания, а следующая повышает качество с помощью разграничения настоящих предсказаний от сгенерированных. Метод может снизить требования по видеопамяти для обучения на доступных GPU: Google Colab T4 (16Gb), а также не требует размеченных данных помимо используемых на этапе SFT. Для анализа качества метода будет использована группа датасетов таких, как gsm8k, ultrachat\_200k.

----


### Датасеты

Для ознакомления ниже расположены данные, которые были использованы для обучения модели Qwen2.5-0.5B-Instruct с использованием адапетров LoRA (r=16, alpha=32)

| Dataset                    |                                       Download                                       |
| :----------------------- |:------------------------------------------------------------------------------------:|
| iter0     | 🤗 [HuggingFace](https://huggingface.co/datasets/MuQianyu/qwen2.5-0.5B_lora16_spin0) |
| iter1 | 🤗 [HuggingFace](https://huggingface.co/datasets/MuQianyu/qwen2.5-0.5B_lora16_spin1) |
| SPIN_iter2      | 🤗 [HuggingFace](https://huggingface.co/datasets/MuQianyu/qwen2.5-0.5B_lora16_spin2) |

----

### Модель

Также предоставляется информация о 

| Model                         |                                  Download                                   |
|:------------------------------|:---------------------------------------------------------------------------:|
| Qwen2.5-0.5B_lora-16_iter-0   | 🤗 [HuggingFace](https://huggingface.co/MuQianyu/qwen2.5-0.5B_lora16_spin0) |
| Qwen2.5-0.5B_lora-16_iter-1   | 🤗 [HuggingFace](https://huggingface.co/MuQianyu/qwen2.5-0.5B_lora16_spin1) |
| Qwen2.5-0.5B_lora-16_iter-2   | 🤗 [HuggingFace](https://huggingface.co/MuQianyu/qwen2.5-0.5B_lora16_spin2) |
