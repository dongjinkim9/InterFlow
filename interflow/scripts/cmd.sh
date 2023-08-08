# train
python main.py \
--parameter-name interflow_scale24.pt \
--scales 2 4 \
--gpu 2;

# test
python main.py \
--parameter-restore-path ./pretrained_models/interflow_scale24.pt \
--scales 2 4 \
--gpu 4 --test;

# generation
python generate_images.py \
--foldername interflow_scale24 \
--parameter ./pretrained_models/interflow_scale24.pt \
--model-scales 2 4 \
--target-scales 2 4 \
--divide-mode 0 \
--gpu 4;