## Training

1. Download restructured RealSR version 2 [dataset](https://1drv.ms/u/c/85cf5b7f538e2007/EQcgjlN_W88ggIWUqQAAAAABQOgqVHg0X7B5NY_u1pD9RA?e=TgwDe2) and place it in `datasets/`

2. To train Interflow with default settings, run
```
python main.py \
--parameter-name interflow_scale24.pt \
--scales 2 4 \
--gpu 0;
```

## Generation

1. Download the pre-trained [model](https://1drv.ms/f/c/85cf5b7f538e2007/EgcgjlN_W88ggIUpRwAAAAABzaBTD730v-ZpO7K8EvABCw?e=ONG0vW) and place it in `pretrained_models/`

2. Generate the images with intermediate degradation levels $\times2$ ~ $\times4$:
```
python generate_images.py \
--foldername interflow_scale24 \
--parameter pretrained_models/interflow_scale24.pt \
--model-scales 2 4 \
--target-scales 2 4 \
--divide-mode 0 \
--gpu 0;
```
