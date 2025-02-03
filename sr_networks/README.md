
## Training
1. Download restructured RealSR version 2 [dataset](https://1drv.ms/u/c/85cf5b7f538e2007/EQcgjlN_W88ggIWUqQAAAAABQOgqVHg0X7B5NY_u1pD9RA?e=TgwDe2) and place it in `./datasets/`

2. (Optional) Download synthesized [dataset](https://1drv.ms/u/c/85cf5b7f538e2007/EQcgjlN_W88ggIWMnQAAAAAB9BH0d1sfTQn3BUxxGwXfig?e=xG3e9r) from Interflow with intermidiate degradation levels
 and place it in `./datasets/`

3. To train SR networks, run
```
python3 main.py \
--GT-paths [TRAIN DATASET PATH] \
--parameter-name [FOLDER NAME TO SAVE THE RESULTS] \
--train-iteration [NUM TRAIN ITERATION] \
--validation-step [VAL ITERATION] \
--test-scale 3.0 \
--model [VDSR|RCAN|KPN|HAN|NLSN|SwinIR|MetaSR|LIIF] \
--gpu [GPU INDEX];
```

Please refer more examples in ``scripts/train.sh``

## Evaluation

1. Download the pre-trained [model](https://1drv.ms/f/c/85cf5b7f538e2007/EgcgjlN_W88ggIUoRwAAAAABX3143bYhvER4tW9Rk9GZAg) and place it in `./pretrained_models/` manually


2. Download test datasets : [RealSR x3](https://1drv.ms/u/c/85cf5b7f538e2007/EQcgjlN_W88ggIWUqQAAAAABQOgqVHg0X7B5NY_u1pD9RA?e=TgwDe2), 
                            [DRealSR x3](https://drive.google.com/drive/folders/16B4ssDaDAsH-kE7LQXY5JOxijq5abqhf)

4. Testing
```
python3 main.py \
--parameter-name [FOLDER NAME TO SAVE THE RESULTS]  \
--parameter-restore-path [PATH TO YOUR CHECKPOINT] \
--test-scale 3.0 \
--model [VDSR|RCAN|KPN|HAN|NLSN|SwinIR|MetaSR|LIIF] \
--test-data [RealSR|DRealSR] \
--gpu [GPU INDEX] \
--test; 
```

Please refer more examples in ``scripts/test.sh``