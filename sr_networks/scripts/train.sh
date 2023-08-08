python3 main.py \
--GT-paths ./datasets/RealSR_v2_ordered/Nikon/Train_divided/2 ./datasets/RealSR_v2_ordered/Nikon/Train_divided/4 \
--parameter-name vdsr_2_4.pt  \
--train-iteration 70000 \
--validation-step 70000 \
--test-scale 3.0 \
--model VDSR \
--gpu 0;

python3 main.py \
--GT-paths ./datasets/RealSR_v2_ordered/Nikon/Train_divided/3 \
--parameter-name vdsr_3.pt  \
--train-iteration 70000 \
--validation-step 70000 \
--test-scale 3.0 \
--model VDSR \
--gpu 0;

python3 main.py \
--GT-paths ./datasets/RealSR_v2_ordered/Nikon/Train_divided/2 ./datasets/RealSR_v2_ordered/Nikon/Train_divided/4 ./datasets/interflow_2_4_images/scale_2.1 ./datasets/interflow_2_4_images/scale_2.2 ./datasets/interflow_2_4_images/scale_2.3 ./datasets/interflow_2_4_images/scale_2.4 ./datasets/interflow_2_4_images/scale_2.5 ./datasets/interflow_2_4_images/scale_2.6 ./datasets/interflow_2_4_images/scale_2.7 ./datasets/interflow_2_4_images/scale_2.8 ./datasets/interflow_2_4_images/scale_2.9 ./datasets/interflow_2_4_images/scale_3.0 ./datasets/interflow_2_4_images/scale_3.1 ./datasets/interflow_2_4_images/scale_3.2 ./datasets/interflow_2_4_images/scale_3.3 ./datasets/interflow_2_4_images/scale_3.4 ./datasets/interflow_2_4_images/scale_3.5 ./datasets/interflow_2_4_images/scale_3.6 ./datasets/interflow_2_4_images/scale_3.7 ./datasets/interflow_2_4_images/scale_3.8 ./datasets/interflow_2_4_images/scale_3.9 \
--parameter-name vdsr_interflow_2__4.pt  \
--train-iteration 70000 \
--validation-step 70000 \
--test-scale 3.0 \
--model VDSR \
--gpu 0;

# Please select one of config in ./configs/arb_train_configs.json
python3 main.py \
--parameter-name LIIF_3.pt  \
--train-iteration 200000 \
--validation-step 200000 \
--test-scale 3.0 \
--model LIIF --encoder EDSR_baseline \
--gpu 0;