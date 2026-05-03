
step=625           
save_dir=inference_outputs/sjd_coco
log_dir=$save_dir/logs                                                                                                                                                                                                                               
mkdir -p "$log_dir"

for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python eval_coco/gen_janus.py \
        --method sjd \
        --begin $((i * step)) --end $(((i + 1) * step)) \
        --prompt_path /jizhicfs/pkuhetu/bht/NRP/datasets/coco2017_val_prompts.json \
        --save_dir    $save_dir \
        --json_key caption --dataset_name COCO \
        --do_decode_image \
        > "${log_dir}/rank_${i}.log" 2>&1
done
wait
