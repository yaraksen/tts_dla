# TTS project
### Aksenov Yaroslav

## Installation guide

Download [model](https://disk.yandex.ru/d/-nriuAv-G-YclQ) to the folder ```final_model```
Create folder ```data```

```shell
pip install -r ./requirements.txt
sh misc/get_misc.sh
python misc/load_stats.py
```

## Launching guide

#### Testing:
   ```shell
   python test.py \
      -c src/train_config.json \
      -wgp waveglow/pretrained_model/waveglow_256channels.pt \
      -r final_model/model_best_tts.pth \
      -o test_res
   ```

#### Training:
   ```shell
   python train.py \
      -c src/train_config.json \
      -wk "YOUR_WANDB_API_KEY"
   ```

#### Тестовые записи
Test audios with different alpha, beta, gamma are available in [test_audios](test_audios) directory