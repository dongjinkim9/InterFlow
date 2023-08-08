
## Training
1. Download restructured RealSR version 2 for training:
```
python download.py --file RealSR_234
```

2. (Optional) To download generated dataset with intermidiate levels from Interflow:
```
python download.py --file RealSR_24_gen
```

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

1. Download the pre-trained [model](https://hyu-my.sharepoint.com/:u:/g/personal/dongjinkim_hanyang_ac_kr/EbQdObqo_pNBoyfMjs10smcBakm9mbdMtFiviemceAaCdA?e=9dgaZ6&download=1) and place it in `./pretrained_models/` manually

OR run:
```
python download.py --file pretrained
```

2. To download RealSR x3 test datasets[[link](https://hyu-my.sharepoint.com/:u:/g/personal/dongjinkim_hanyang_ac_kr/EYF4Xt2V-rFEoLzbaUq6LVABByEe-Yc55-xFhIPhHkfK6A?e=LQc5Nq&download=1)], run 
```
python download.py --file RealSR_234
```

3. To download DRealSR x3 test datasets [[offical link](https://drive.google.com/drive/folders/16B4ssDaDAsH-kE7LQXY5JOxijq5abqhf)], run 
```
python download.py --file DRealSR_3_test
```


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