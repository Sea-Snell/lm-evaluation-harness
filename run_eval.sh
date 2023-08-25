
# 8/25/23

conda activate torch_install

# # 1B_v1_data
# for STEP in 9800 5600 1400
# do
#         echo "1B_v1_data ($STEP) starting ..."
#         export BATCH_SIZE=16
#         export MODEL_PATH=/shared/csnell/data_study/1B_v1_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/1B_v1_data/$STEP/evals2
#         export CUDA_VISIBLE_DEVICES=0,2,3,4,6,7,8,9
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hendrycksTest-*,triviaqa' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "1B_v1_data ($STEP) done."
# done

# 1B_v2_data
for STEP in 9800 5600 1400
do
        echo "1B_v2_data ($STEP) starting ..."
        export BATCH_SIZE=16
        export MODEL_PATH=/shared/csnell/data_study/1B_v2_data/$STEP/pytorch
        export OUTPUT_PATH=/shared/csnell/data_study/1B_v2_data/$STEP/evals2
        export CUDA_VISIBLE_DEVICES=0,2,3,4,6,7,8,9
        mkdir $OUTPUT_PATH
        python main.py \
                --model hf-causal-experimental \
                --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
                --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
                --num_fewshot=0 \
                --device cuda \
                --output_path $OUTPUT_PATH/0shot.json \
                --batch_size $BATCH_SIZE \
                --no_cache
        python main.py \
                --model hf-causal-experimental \
                --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
                --tasks 'hendrycksTest-*,triviaqa' \
                --num_fewshot=5 \
                --device cuda \
                --output_path $OUTPUT_PATH/5shot.json \
                --batch_size $BATCH_SIZE \
                --no_cache
        echo "1B_v2_data ($STEP) done."
done

# # 3B_v1_data
# for STEP in 35200 28800 22400
# do
#         echo "3B_v1_data ($STEP) starting ..."
#         export BATCH_SIZE=2
#         export MODEL_PATH=/shared/csnell/data_study/3B_v1_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/3B_v1_data/$STEP/evals2
#         export CUDA_VISIBLE_DEVICES=1,5
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hendrycksTest-*,triviaqa' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "3B_v1_data ($STEP) done."
# done

# # 3B_v2_data
# for STEP in 35200 28800 22400
# do
#         echo "3B_v2_data ($STEP) starting ..."
#         export STEP=44000
#         export BATCH_SIZE=2
#         export MODEL_PATH=/shared/csnell/data_study/3B_v2_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/3B_v2_data/$STEP/evals2
#         export CUDA_VISIBLE_DEVICES=1,5
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hendrycksTest-*,triviaqa' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "3B_v2_data ($STEP) done."
# done

# 8/23/23

# conda activate torch_install

# # 1B_v1_data
# echo "1B_v1_data"
# export STEP=14000
# export BATCH_SIZE=32
# export MODEL_PATH=/shared/csnell/data_study/1B_v1_data/$STEP/pytorch
# export OUTPUT_PATH=/shared/csnell/data_study/1B_v1_data/$STEP/evals2
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# mkdir $OUTPUT_PATH
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#         --num_fewshot=0 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/0shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hendrycksTest-*,triviaqa' \
#         --num_fewshot=5 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/5shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache

# # 1B_v2_data
# echo "1B_v2_data"
# export STEP=14000
# export BATCH_SIZE=32
# export MODEL_PATH=/shared/csnell/data_study/1B_v2_data/$STEP/pytorch
# export OUTPUT_PATH=/shared/csnell/data_study/1B_v2_data/$STEP/evals2
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# mkdir $OUTPUT_PATH
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#         --num_fewshot=0 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/0shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hendrycksTest-*,triviaqa' \
#         --num_fewshot=5 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/5shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache

# # 3B_v1_data
# echo "3B_v1_data"
# export STEP=44000
# export BATCH_SIZE=8
# export MODEL_PATH=/shared/csnell/data_study/3B_v1_data/$STEP/pytorch
# export OUTPUT_PATH=/shared/csnell/data_study/3B_v1_data/$STEP/evals2
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# mkdir $OUTPUT_PATH
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#         --num_fewshot=0 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/0shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hendrycksTest-*,triviaqa' \
#         --num_fewshot=5 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/5shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache

# # 3B_v2_data
# echo "3B_v2_data"
# export STEP=44000
# export BATCH_SIZE=8
# export MODEL_PATH=/shared/csnell/data_study/3B_v2_data/$STEP/pytorch
# export OUTPUT_PATH=/shared/csnell/data_study/3B_v2_data/$STEP/evals2
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# mkdir $OUTPUT_PATH
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#         --num_fewshot=0 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/0shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hendrycksTest-*,triviaqa' \
#         --num_fewshot=5 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/5shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache

# 8/22/23

# conda activate torch_install

# for STEP in 9800 5600 1400
# do
#         echo "1B_v1_data ($STEP)"
#         export BATCH_SIZE=32
#         export MODEL_PATH=/shared/csnell/data_study/1B_v1_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/1B_v1_data/$STEP/evals
#         export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hendrycksTest-*,triviaqa' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
# done

# for STEP in 9800 5600 1400
# do
#         echo "1B_v2_data ($STEP)"
#         export BATCH_SIZE=32
#         export MODEL_PATH=/shared/csnell/data_study/1B_v2_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/1B_v2_data/$STEP/evals
#         export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hendrycksTest-*,triviaqa' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
# done

# for STEP in 35200 28800 22400
# do
#         echo "3B_v1_data ($STEP)"
#         export BATCH_SIZE=8
#         export MODEL_PATH=/shared/csnell/data_study/3B_v1_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/3B_v1_data/$STEP/evals
#         export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hendrycksTest-*,triviaqa' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
# done

# for STEP in 30800 17600 4400
# do
#         echo "3B_v2_data ($STEP)"
#         export BATCH_SIZE=8
#         export MODEL_PATH=/shared/csnell/data_study/3B_v2_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/3B_v2_data/$STEP/evals
#         export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hendrycksTest-*,triviaqa' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
# done

# old

# conda activate torch_install

# # 1B_v2_data
# echo "1B_v2_data"
# export BATCH_SIZE=32
# export MODEL_PATH=/shared/csnell/data_study/1B_v2_data/pytorch
# export OUTPUT_PATH=/shared/csnell/data_study/1B_v2_data/evals
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# mkdir $OUTPUT_PATH
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#         --num_fewshot=0 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/0shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hendrycksTest-*,triviaqa' \
#         --num_fewshot=5 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/5shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache

# # 3B_v1_data
# echo "3B_v1_data"
# export BATCH_SIZE=8
# export MODEL_PATH=/shared/csnell/data_study/3B_v1_data/pytorch
# export OUTPUT_PATH=/shared/csnell/data_study/3B_v1_data/evals
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# mkdir $OUTPUT_PATH
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#         --num_fewshot=0 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/0shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hendrycksTest-*,triviaqa' \
#         --num_fewshot=5 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/5shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache

# # 3B_v2_data
# echo "3B_v2_data"
# export BATCH_SIZE=8
# export MODEL_PATH=/shared/csnell/data_study/3B_v2_data/pytorch
# export OUTPUT_PATH=/shared/csnell/data_study/3B_v2_data/evals
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# mkdir $OUTPUT_PATH
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#         --num_fewshot=0 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/0shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hendrycksTest-*,triviaqa' \
#         --num_fewshot=5 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/5shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache


# 1B_v1_data
# echo "1B_v1_data"
# export BATCH_SIZE=32
# export MODEL_PATH=/shared/csnell/data_study/1B_v1_data/pytorch
# export OUTPUT_PATH=/shared/csnell/data_study/1B_v1_data/evals
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# mkdir $OUTPUT_PATH
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#         --num_fewshot=0 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/0shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hendrycksTest-*,triviaqa' \
#         --num_fewshot=5 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/5shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache
