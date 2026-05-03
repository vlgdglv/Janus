
step=625           
gsd_G=3
save_dir=inference_outputs/gsd_coco_${gsd_G}
log_dir=$save_dir/logs                                                                                                                                                                                                                               
mkdir -p "$log_dir"

for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python eval_coco/gen_janus.py \
        --method gsd --gsd_G ${gsd_G} \
        --begin $((i * step)) --end $(((i + 1) * step)) \
        --prompt_path /jizhicfs/pkuhetu/bht/NRP/datasets/coco2017_val_prompts.json \
        --save_dir    $save_dir \
        --json_key caption --dataset_name COCO \
        --do_decode_image \
        > "${log_dir}/rank_${i}.log" 2>&1
done
wait
