export PYTHONPATH="./:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=5

PRETRAINED_DIR="./models/pretrained_dorm"
DATE_DIR="data"


index=1
learning_rate=75e-6
pinyin_weight=0.2
add_pinyin_mask="True"
add_pinyin_loss="True"
use_kl="True"
kl_weight=1.5
second_loss_weight=0.97
gradient_accumulation_steps=1
random_seed=42
OUTPUT_DIR="./models/finetuning_dorm_${index}"

year=15
python -m torch.distributed.launch --master_port=24679 --nproc_per_node=1 src/dorm_finetune.py \
    --model_type dorm \
    --model_name_or_path $PRETRAINED_DIR \
    --image_model_type 0 \
    --output_dir $OUTPUT_DIR  \
    --do_train --do_eval --do_predict \
    --data_dir $DATE_DIR \
    --train_file trainall.times2_pinyin2.pkl \
    --dev_file "test.sighan${year}_pinyin2.pkl" \
    --dev_label_file "test.sighan${year}.lbl.tsv" \
    --predict_file "test.sighan${year}_pinyin2.pkl" \
    --predict_label_file "test.sighan${year}.lbl.tsv" \
    --order_metric sent-correct-f1  \
    --max_seq_length 512 \
    --metric_reverse  \
    --num_save_ckpts 3 \
    --remove_unused_ckpts  \
    --per_gpu_train_batch_size 5 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --per_gpu_eval_batch_size 1  \
    --learning_rate $learning_rate \
    --num_train_epochs 1  \
    --seed $random_seed \
    --warmup_steps 1000  \
    --overwrite_output_dir \
    --add_pinyin_mask \
    --add_pinyin_loss \
    --pinyin_weight $pinyin_weight \
    --use_kl \
    --kl_weight $kl_weight \
    --second_loss_weight $second_loss_weight \
    --save_steps 2000 \
    --eval_all_checkpoints


