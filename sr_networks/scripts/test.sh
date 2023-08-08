# test RealSR x3
python3 main.py \
--parameter-name vdsr_ours.pt  \
--parameter-restore-path ./pretrained_models/sr_networks/vdsr_ours.pt \
--test-scale 3.0 \
--model VDSR \
--test \
--gpu 0; 

python3 main.py \
--parameter-name liif_ours.pt  \
--parameter-restore-path ./pretrained_models/sr_networks/liif_ours.pt \
--test-scale 3.0 \
--model LIIF --encoder EDSR_baseline \
--test \
--gpu 0; 

# test DRealSR x3
python3 main.py \
--parameter-name liif_ours.pt  \
--parameter-restore-path ./pretrained_models/sr_networks/liif_ours.pt \
--GT-paths ./datasets/DRealsr_x3_test \
--test-scale 3.0 \
--model LIIF --encoder EDSR_baseline \
--test-data DRealSR \
--test \
--gpu 0; 