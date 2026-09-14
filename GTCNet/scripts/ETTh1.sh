export CUDA_VISIBLE_DEVICES=1

model_name=GTCNet
folder_name="${model_name}"

echo "start training..."

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/$folder_name" ]; then
    mkdir ./logs/$folder_name
fi

seq_len=96
label_len=48

# ===================== pred_len=96 =====================
pred_len=96
d_model=64
dropout=0.2
linformer_rank=64
python -u run.py \
--is_training 1 \
--model $model_name \
--data ETTh1 \
--root_path ./dataset \
--data_path ETTh1.csv \
--features M \
--seq_len $seq_len \
--label_len $label_len \
--pred_len $pred_len \
--enc_in 7 \
--patch_len 16 \
--stride 8 \
--d_model $d_model \
--n_heads 8 \
--n_layers 1 \
--temperature 0.3 \
--dropout $dropout \
--mlp_ratio 2 \
--linformer_rank $linformer_rank \
--lambda_gate 1e-3 \
--batch_size 2048 \
--learning_rate 0.0005 \
--train_epochs 100 \
--patience 20 \
--lradj sigmoid \
--revin 1 \
--des Exp \
--itr 1 > logs/${folder_name}/${model_name}_ETTh1_${seq_len}_${pred_len}.log 2>&1

# ===================== pred_len=192 =====================
pred_len=192
d_model=96
dropout=0.1
linformer_rank=64
python -u run.py \
--is_training 1 \
--model $model_name \
--data ETTh1 \
--root_path ./dataset \
--data_path ETTh1.csv \
--features M \
--seq_len $seq_len \
--label_len $label_len \
--pred_len $pred_len \
--enc_in 7 \
--patch_len 16 \
--stride 8 \
--d_model $d_model \
--n_heads 8 \
--n_layers 1 \
--temperature 0.3 \
--dropout $dropout \
--mlp_ratio 2 \
--linformer_rank $linformer_rank \
--lambda_gate 1e-3 \
--batch_size 2048 \
--learning_rate 0.0005 \
--train_epochs 100 \
--patience 20 \
--lradj sigmoid \
--revin 1 \
--des Exp \
--itr 1 > logs/${folder_name}/${model_name}_ETTh1_${seq_len}_${pred_len}.log 2>&1

# ===================== pred_len=336 =====================
pred_len=336
d_model=128
dropout=0.1
linformer_rank=64
python -u run.py \
--is_training 1 \
--model $model_name \
--data ETTh1 \
--root_path ./dataset \
--data_path ETTh1.csv \
--features M \
--seq_len $seq_len \
--label_len $label_len \
--pred_len $pred_len \
--enc_in 7 \
--patch_len 16 \
--stride 8 \
--d_model $d_model \
--n_heads 8 \
--n_layers 1 \
--temperature 0.3 \
--dropout $dropout \
--mlp_ratio 2 \
--linformer_rank $linformer_rank \
--lambda_gate 1e-3 \
--batch_size 2048 \
--learning_rate 0.0005 \
--train_epochs 100 \
--patience 20 \
--lradj sigmoid \
--revin 1 \
--des Exp \
--itr 1 > logs/${folder_name}/${model_name}_ETTh1_${seq_len}_${pred_len}.log 2>&1

# ===================== pred_len=720 =====================
pred_len=720
d_model=96
dropout=0.1
linformer_rank=16
python -u run.py \
--is_training 1 \
--model $model_name \
--data ETTh1 \
--root_path ./dataset \
--data_path ETTh1.csv \
--features M \
--seq_len $seq_len \
--label_len $label_len \
--pred_len $pred_len \
--enc_in 7 \
--patch_len 16 \
--stride 8 \
--d_model $d_model \
--n_heads 8 \
--n_layers 1 \
--temperature 0.3 \
--dropout $dropout \
--mlp_ratio 2 \
--linformer_rank $linformer_rank \
--lambda_gate 1e-3 \
--batch_size 2048 \
--learning_rate 0.0005 \
--train_epochs 100 \
--patience 20 \
--lradj sigmoid \
--revin 1 \
--des Exp \
--itr 1 > logs/${folder_name}/${model_name}_ETTh1_${seq_len}_${pred_len}.log 2>&1