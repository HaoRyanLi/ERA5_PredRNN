export CUDA_VISIBLE_DEVICES=0,1,2
cd ..
base_path="/scratch/09012/haoli1/ERA5/dataset/"
train_data_paths=""
for file in /scratch/09012/haoli1/ERA5/dataset/*; do
    train_data_paths+="${file},"
done

# Remove the trailing comma
train_data_paths=${train_data_paths%,}

python -u run1.py \
    --is_training 1 \
    --device cuda:0 \
    --mem_alloc_conf 0 \
    --dataset_name mnist \
    --train_data_paths ${train_data_paths} \
    --valid_data_paths /scratch/09012/haoli1/ERA5/val_dataset/era5_train_09012021_3_24hr.npz \
    --extra_var_paths /scratch/09012/haoli1/ERA5/exta_var.npz\
    --save_dir /work/09012/haoli1/ls6/PredRNN_checkpoints/ \
    --gen_frm_dir /work/09012/haoli1/ls6/PredRNN_checkpoints/ \
    --model_name predrnn_v2 \
    --reverse_input 0 \
    --test_batch_size 15\
    --add_geopential 0 \
    --add_land 0 \
    --add_latitude 0 \
    --is_WV 2 \
    --press_constraint 1 \
    --center_enhance 0 \
    --patch_size 40 \
    --weighted_loss 1 \
    --upload_run 0 \
    --layer_need_enhance 1 \
    --find_max False \
    --multiply 2 \
    --img_height 720 \
    --img_width 1440 \
    --use_weight 0 \
    --layer_weight 20 \
    --img_channel 3 \
    --img_layers 0,1,2 \
    --input_length 24 \
    --total_length 48 \
    --num_hidden 512,512,512,512 \
    --skip_time 1 \
    --wavelet db1 \
    --filter_size 5 \
    --stride 1 \
    --layer_norm 1 \
    --decouple_beta 0.05 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 2e-4 \
    --batch_size 3 \
    --test_batch_size 9 \
    --max_iterations 10000 \
    --display_interval 1000 \
    --test_interval 10 \
    --snapshot_interval 1000 \
    --conv_on_input 0 \
    --res_on_conv 0 \
    --curr_best_mse 0.025 \
    --save_best_name wv0_pc0 \
    --pretrained_model /work/09012/haoli1/ls6/PredRNN_checkpoints/ \
    --pretrained_model_name model_init.ckpt \
