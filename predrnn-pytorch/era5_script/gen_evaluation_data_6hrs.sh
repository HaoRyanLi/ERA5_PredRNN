export CUDA_VISIBLE_DEVICES=0,1,2
cd ..
base_path="/scratch/09012/haoli1/ERA5/dataset/"
train_data_paths=""
for file in /scratch/09012/haoli1/ERA5/dataset/*; do
    train_data_paths+="${file},"
done
load_path="/work/09012/haoli1/ls6/PredRNN_checkpoints/"

# Remove the trailing comma
train_data_paths=${train_data_paths%,}

python -u run1.py \
    --is_training 0 \
    --concurent_step 2 \
    --device cuda:0 \
    --dataset_name mnist \
    --train_data_paths ${train_data_paths} \
    --valid_data_paths /scratch/09012/haoli1/ERA5/val_dataset_6hrs/era5_train_2022_3_24hr.npz \
    --save_data_names 2022_3 \
    --extra_var_paths /scratch/09012/haoli1/ERA5/exta_var.npz\
    --gen_frm_dir ${load_path} \
    --gen_data_dir /scratch/09012/haoli1/ERA5/evaluation/ \
    --save_test_result 1 \
    --model_name predrnn_v2 \
    --reverse_input 0 \
    --batch_size 2 \
    --test_batch_size 6\
    --gpu_num 3\
    --is_WV 1 \
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
    --layer_weight 10,10,10,10,20,20 \
    --img_channel 3 \
    --img_layers 0,1,2 \
    --input_length 28 \
    --total_length 56 \
    --time_step 6hrs \
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
    --lr 1e-4 \
    --max_iterations 3 \
    --display_interval 10 \
    --test_interval 200 \
    --snapshot_interval 200 \
    --conv_on_input 0 \
    --res_on_conv 0 \
    --multiply 1.1 \
    --pretrained_model ${load_path} \
    --pretrained_model_name model_wv1_pc1.ckpt \
    # --pretrained_model_name model_test_schedule_sampling_DynaMean.ckpt \

#cp /scratch/network/hvtran/era5/predrnn-pytorch/checkpoints/era5_predrnn/model.ckpt-1000 /home/hvtran/
# /work/09012/haoli1/ls6/ERA5/era5_train_0921002022_3_24hr.npz 
