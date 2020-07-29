python3 dlrm_main.py --batch_size=55296 --bfloat16_grads_all_reduce --nobinarylog --${INTERNAL}_num_eigen_threads=40 --${INTERNAL}_num_operation_threads=40 --${INTERNAL}_port=14832 --${INTERNAL}_rpc_layer=rpc2 --${INTERNAL}_run_locally --census_cpu_accounting_enabled --data_dir=/export/ssd/mlperf_data/dlrm --decay_start_step=49315 --decay_steps=27772 --dim_embed=128 --noenable_profiling --noenable_summary --eval_batch_size=55296 --eval_steps=1616 --learning_rate=0.887 --lr_warmup_steps=2750 --master=mvbn1:14678 --mlp_bottom=512,256,128 --mlp_top=1024,1024,512,256,1 --num_dense_features=13 --num_tables_in_ec=13 --num_tpu_shards=128 --optimizer=sgd --nopipeline_execution --replicas_per_host=4 --norestore_checkpoint --rpclog=-1 --nosave_checkpoint --sleep_after_init=60 --steps_between_evals=3793 --train_steps=75860 --use_batched_tfrecords --nouse_cached_data --nouse_synthetic_data --vocab_sizes_embed=39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36