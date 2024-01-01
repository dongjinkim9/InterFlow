## Training

1. Download restructured RealSR version 2 dataset for training and generation:
```
python download_dataset.py
```

2. To train Interflow with default settings, run
```
python main.py \
--parameter-name interflow_scale24.pt \
--scales 2 4 \
--gpu 0;
```

## Generation

1. Download the pre-trained [model](https://onedrive.live.com/download?resid=85CF5B7F538E2007%2123852&authkey=!AIrAVFRUcCHoYoM) and place it in `./pretrained_models/`

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
