
# 2/12/24

conda activate torch_install

# 3B_v1
for STEP in 2500 5000 7500
do
        echo "3B_v1 ($STEP) starting ..."
        export BATCH_SIZE=64
        export MODEL_PATH=/shared/csnell/openllama/3B_v1/$STEP/pytorch
        export OUTPUT_PATH=/shared/csnell/openllama/3B_v1/$STEP/early_checkpoint_evals
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        mkdir $OUTPUT_PATH
        python main.py \
                --model hf-causal-experimental \
                --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
                --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,winogrande,piqa,arc_easy,arc_challenge,hellaswag,boolq,openbookqa,race,record' \
                --num_fewshot=0 \
                --device cuda \
                --output_path $OUTPUT_PATH/0shot.json \
                --batch_size $BATCH_SIZE \
                --no_cache
        python main.py \
                --model hf-causal-experimental \
                --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
                --tasks 'lambada_openai,lambada_openai_cloze,squad2,winogrande,openbookqa,boolq,triviaqa' \
                --num_fewshot=5 \
                --device cuda \
                --output_path $OUTPUT_PATH/5shot.json \
                --batch_size $BATCH_SIZE \
                --no_cache
        echo "3B_v1 ($STEP) done."
done

# 7B_v1
for STEP in 2500 5000 7500
do
        echo "7B_v1 ($STEP) starting ..."
        export BATCH_SIZE=32
        export MODEL_PATH=/shared/csnell/openllama/7B_v1/$STEP/pytorch
        export OUTPUT_PATH=/shared/csnell/openllama/7B_v1/$STEP/early_checkpoint_evals
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        mkdir $OUTPUT_PATH
        python main.py \
                --model hf-causal-experimental \
                --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
                --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,winogrande,piqa,arc_easy,arc_challenge,hellaswag,boolq,openbookqa,race,record' \
                --num_fewshot=0 \
                --device cuda \
                --output_path $OUTPUT_PATH/0shot.json \
                --batch_size $BATCH_SIZE \
                --no_cache
        python main.py \
                --model hf-causal-experimental \
                --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
                --tasks 'lambada_openai,lambada_openai_cloze,squad2,winogrande,openbookqa,boolq,triviaqa' \
                --num_fewshot=5 \
                --device cuda \
                --output_path $OUTPUT_PATH/5shot.json \
                --batch_size $BATCH_SIZE \
                --no_cache
        echo "7B_v1 ($STEP) done."
done

# 13B_v1
for STEP in 5000 10000 15000
do
        echo "13B_v1 ($STEP) starting ..."
        export BATCH_SIZE=16
        export MODEL_PATH=/shared/csnell/openllama/13B_v1/$STEP/pytorch
        export OUTPUT_PATH=/shared/csnell/openllama/13B_v1/$STEP/early_checkpoint_evals
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        mkdir $OUTPUT_PATH
        python main.py \
                --model hf-causal-experimental \
                --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
                --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,winogrande,piqa,arc_easy,arc_challenge,hellaswag,boolq,openbookqa,race,record' \
                --num_fewshot=0 \
                --device cuda \
                --output_path $OUTPUT_PATH/0shot.json \
                --batch_size $BATCH_SIZE \
                --no_cache
        python main.py \
                --model hf-causal-experimental \
                --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
                --tasks 'lambada_openai,lambada_openai_cloze,squad2,winogrande,openbookqa,boolq,triviaqa' \
                --num_fewshot=5 \
                --device cuda \
                --output_path $OUTPUT_PATH/5shot.json \
                --batch_size $BATCH_SIZE \
                --no_cache
        echo "13B_v1 ($STEP) done."
done

# 11/8/23

# # openllama2 logloss study evals

<<<<<<< HEAD
# 3B_v2
for STEP in 20000 100000 200000 300000 400000 460000
do
        echo "3B_v2 ($STEP) starting ..."
        export BATCH_SIZE=32
        export MODEL_PATH=/shared/csnell/openllama/3B_v2/$STEP/pytorch
        export OUTPUT_PATH=/shared/csnell/openllama/3B_v2/$STEP/evals_1
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
        mkdir $OUTPUT_PATH
        python main.py \
                --model hf-causal-experimental \
                --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
                --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race,lambada_openai,lambada_openai_cloze,copa,cola,squad2,wikitext,bigbench_bb_data_study-*,bigbench_bb_hard-*,bigbench_bb_lite-*' \
                --num_fewshot=0 \
                --device cuda \
                --output_path $OUTPUT_PATH/0shot.json \
                --batch_size $BATCH_SIZE \
                --no_cache
        python main.py \
                --model hf-causal-experimental \
                --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
                --tasks 'hendrycksTest-*,triviaqa,lambada_openai,lambada_openai_cloze,cola,squad2,winogrande,openbookqa,boolq,rte,wic' \
                --num_fewshot=5 \
                --device cuda \
                --output_path $OUTPUT_PATH/5shot.json \
                --batch_size $BATCH_SIZE \
                --no_cache
        echo "3B_v2 ($STEP) done."
done

# 7B_v2
for STEP in 20000 100000 200000 300000 400000 460000
do
        echo "7B_v2 ($STEP) starting ..."
        export BATCH_SIZE=32
        export MODEL_PATH=/shared/csnell/openllama/7B_v2/$STEP/pytorch
        export OUTPUT_PATH=/shared/csnell/openllama/7B_v2/$STEP/evals_1
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
        mkdir $OUTPUT_PATH
        python main.py \
                --model hf-causal-experimental \
                --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
                --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race,lambada_openai,lambada_openai_cloze,copa,cola,squad2,wikitext,bigbench_bb_data_study-*,bigbench_bb_hard-*,bigbench_bb_lite-*' \
                --num_fewshot=0 \
                --device cuda \
                --output_path $OUTPUT_PATH/0shot.json \
                --batch_size $BATCH_SIZE \
                --no_cache
        python main.py \
                --model hf-causal-experimental \
                --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
                --tasks 'hendrycksTest-*,triviaqa,lambada_openai,lambada_openai_cloze,cola,squad2,winogrande,openbookqa,boolq,rte,wic' \
                --num_fewshot=5 \
                --device cuda \
                --output_path $OUTPUT_PATH/5shot.json \
                --batch_size $BATCH_SIZE \
                --no_cache
        echo "7B_v2 ($STEP) done."
done
=======
# # 3B_v2
# for STEP in 20000 100000 200000 300000 400000 460000
# do
#         echo "3B_v2 ($STEP) starting ..."
#         export BATCH_SIZE=32
#         export MODEL_PATH=/shared/csnell/openllama/3B_v2/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/3B_v2/$STEP/evals_1
#         export CUDA_VISIBLE_DEVICES=3,4,6,7,8,9
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race,lambada_openai,lambada_openai_cloze,copa,cola,squad2,wikitext,bigbench_bb_data_study-*,bigbench_bb_hard-*,bigbench_bb_lite-*' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hendrycksTest-*,triviaqa,lambada_openai,lambada_openai_cloze,cola,squad2,winogrande,openbookqa,boolq,rte,wic' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "3B_v2 ($STEP) done."
# done

# # 7B_v2
# for STEP in 20000 100000 200000 300000 400000 460000
# do
#         echo "7B_v2 ($STEP) starting ..."
#         export BATCH_SIZE=32
#         export MODEL_PATH=/shared/csnell/openllama/7B_v2/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/7B_v2/$STEP/evals_1
#         export CUDA_VISIBLE_DEVICES=3,4,6,7,8,9
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race,lambada_openai,lambada_openai_cloze,copa,cola,squad2,wikitext,bigbench_bb_data_study-*,bigbench_bb_hard-*,bigbench_bb_lite-*' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hendrycksTest-*,triviaqa,lambada_openai,lambada_openai_cloze,cola,squad2,winogrande,openbookqa,boolq,rte,wic' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "7B_v2 ($STEP) done."
# done
>>>>>>> e7e06326c4a3e884d36f7b6742a3d05bd92c454d



# 9/22/23

# logloss study evals

# conda activate torch_install

# for STEP in 10000
# do
#         echo "3B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=32
#         export MODEL_PATH=/shared/csnell/openllama/3B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/3B_v1/$STEP/evals_2_additional
#         export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
        # python main.py \
        #         --model hf-causal-experimental \
        #         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
        #         --tasks 'lambada_openai,lambada_openai_cloze,copa,cola,squad2,wikitext,bigbench_bb_data_study-*,bigbench_bb_hard-*,bigbench_bb_lite-*' \
        #         --num_fewshot=0 \
        #         --device cuda \
        #         --output_path $OUTPUT_PATH/0shot.json \
        #         --batch_size $BATCH_SIZE \
        #         --no_cache
        # python main.py \
        #         --model hf-causal-experimental \
        #         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
        #         --tasks 'lambada_openai,lambada_openai_cloze,cola,squad2,winogrande,openbookqa,boolq,rte,wic' \
        #         --num_fewshot=5 \
        #         --device cuda \
        #         --output_path $OUTPUT_PATH/5shot.json \
        #         --batch_size $BATCH_SIZE \
        #         --no_cache
#         echo "3B_v1 ($STEP) done."
# done

# 3B_v1
# for STEP in 50000 100000 150000 200000 250000
# do
#         echo "3B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=32
#         export MODEL_PATH=/shared/csnell/openllama/3B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/3B_v1/$STEP/evals_2_additional
#         export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,copa,cola,squad2,wikitext,bigbench_bb_data_study-*,bigbench_bb_hard-*,bigbench_bb_lite-*' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,cola,squad2,winogrande,openbookqa,boolq,rte,wic' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "3B_v1 ($STEP) done."
# done

# 7B_v1
# for STEP in 10000 50000 100000 150000 200000 250000
# do
#         echo "7B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=16
#         export MODEL_PATH=/shared/csnell/openllama/7B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/7B_v1/$STEP/evals_2_additional
#         export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,copa,cola,squad2,wikitext,bigbench_bb_data_study-*,bigbench_bb_hard-*,bigbench_bb_lite-*' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,cola,squad2,winogrande,openbookqa,boolq,rte,wic' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "7B_v1 ($STEP) done."
# done

# for STEP in 100000 150000 200000 250000
# do
#         echo "7B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=32
#         export MODEL_PATH=/shared/csnell/openllama/7B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/7B_v1/$STEP/evals_2_additional
#         export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,copa,cola,squad2,wikitext,bigbench_bb_data_study-*,bigbench_bb_hard-*,bigbench_bb_lite-*' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,cola,squad2,winogrande,openbookqa,boolq,rte,wic' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "7B_v1 ($STEP) done."
# done

# 13B_v1
# for STEP in 20000 100000 200000 300000 400000 500000
# do
#         echo "13B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=8
#         export MODEL_PATH=/shared/csnell/openllama/13B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/13B_v1/$STEP/evals_2_additional
#         export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,copa,cola,squad2,wikitext,bigbench_bb_data_study-*,bigbench_bb_hard-*,bigbench_bb_lite-*' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,cola,squad2,winogrande,openbookqa,boolq,rte,wic' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "13B_v1 ($STEP) done."
# done

# for STEP in 300000 400000 500000
# do
#         echo "13B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=24
#         export MODEL_PATH=/shared/csnell/openllama/13B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/13B_v1/$STEP/evals_2_additional
#         export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,copa,cola,squad2,wikitext,bigbench_bb_data_study-*,bigbench_bb_hard-*,bigbench_bb_lite-*' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,cola,squad2,winogrande,openbookqa,boolq,rte,wic' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "13B_v1 ($STEP) done."
# done

# data_study_baseline_evals

# conda activate torch_install

# # official_openllama_3B_v1_data
# echo "official_openllama_3B_v1_data last starting ..."
# export BATCH_SIZE=128
# export MODEL_PATH=openlm-research/open_llama_3b
# export OUTPUT_PATH=/shared/csnell/data_study/official_openllama_3B_v1_data/last/evals2
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# mkdir $OUTPUT_PATH
# # python main.py \
# #         --model hf-causal-experimental \
# #         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
# #         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race,lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
# #         --num_fewshot=0 \
# #         --device cuda \
# #         --output_path $OUTPUT_PATH/0shot.json \
# #         --batch_size $BATCH_SIZE \
# #         --no_cache
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hendrycksTest-*,triviaqa,bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola' \
#         --num_fewshot=5 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/5shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache
# echo "official_openllama_3B_v1_data last done."

# # official_openllama_3B_v2_data
# echo "official_openllama_3B_v2_data last starting ..."
# export BATCH_SIZE=128
# export MODEL_PATH=openlm-research/open_llama_3b_v2
# export OUTPUT_PATH=/shared/csnell/data_study/official_openllama_3B_v2_data/last/evals2
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# mkdir $OUTPUT_PATH
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race,lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#         --num_fewshot=0 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/0shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hendrycksTest-*,triviaqa,bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola' \
#         --num_fewshot=5 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/5shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache
# echo "official_openllama_3B_v2_data last done."



# 9/20/23

# log_loss_study_evals

# conda activate torch_install

# # 3B_v1
# for STEP in 10000 50000 100000 150000 200000 250000
# do
#         echo "3B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=8
#         export MODEL_PATH=/shared/csnell/openllama/3B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/3B_v1/$STEP/evals_2_additional
#         export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext,bigbench_bb_full-*' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'cola,bigbench_bb_full-*' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "3B_v1 ($STEP) done."
# done

# # 7B_v1
# for STEP in 10000 50000 100000 150000 200000 250000
# do
#         echo "7B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=8
#         export MODEL_PATH=/shared/csnell/openllama/7B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/7B_v1/$STEP/evals_2_additional
#         export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext,bigbench_bb_full-*' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'cola,bigbench_bb_full-*' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "7B_v1 ($STEP) done."
# done

# # 13B_v1
# for STEP in 20000 100000 200000 300000 400000 500000
# do
#         echo "13B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=4
#         export MODEL_PATH=/shared/csnell/openllama/13B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/13B_v1/$STEP/evals_2_additional
#         export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext,bigbench_bb_full-*' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'cola,bigbench_bb_full-*' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "13B_v1 ($STEP) done."
# done



# data_study_baseline_evals

# conda activate torch_install

# official_openllama_7B_v1_data
# echo "official_openllama_7B_v1_data last starting ..."
# export BATCH_SIZE=32
# export MODEL_PATH=openlm-research/open_llama_7b
# export OUTPUT_PATH=/shared/csnell/data_study/official_openllama_7B_v1_data/last/evals2
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# mkdir $OUTPUT_PATH
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race,lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#         --num_fewshot=0 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/0shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hendrycksTest-*,triviaqa,bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola' \
#         --num_fewshot=5 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/5shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache
# echo "official_openllama_7B_v1_data last done."

# official_openllama_7B_v2_data
# echo "official_openllama_7B_v2_data last starting ..."
# export BATCH_SIZE=32
# export MODEL_PATH=openlm-research/open_llama_7b_v2
# export OUTPUT_PATH=/shared/csnell/data_study/official_openllama_7B_v2_data/last/evals2
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# mkdir $OUTPUT_PATH
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race,lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#         --num_fewshot=0 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/0shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache
# python main.py \
#         --model hf-causal-experimental \
#         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         --tasks 'hendrycksTest-*,triviaqa,bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola' \
#         --num_fewshot=5 \
#         --device cuda \
#         --output_path $OUTPUT_PATH/5shot.json \
#         --batch_size $BATCH_SIZE \
#         --no_cache
# echo "official_openllama_7B_v2_data last done."



# 9/13/23

# data_study_additional_evals

# conda activate torch_install

# # 1B_v1_data
# for STEP in 14000
# do
#         echo "1B_v1_data ($STEP) starting ..."
#         export BATCH_SIZE=128
#         export MODEL_PATH=/shared/csnell/data_study/1B_v1_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/1B_v1_data/$STEP/evals2_additional
#         export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "1B_v1_data ($STEP) done."
# done

# # 1B_v2_data
# for STEP in 14000
# do
#         echo "1B_v2_data ($STEP) starting ..."
#         export BATCH_SIZE=128
#         export MODEL_PATH=/shared/csnell/data_study/1B_v2_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/1B_v2_data/$STEP/evals2_additional
#         export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "1B_v2_data ($STEP) done."
# done

# 9/13/23

# data_study_additional_evals

# conda activate torch_install

# # 7B_v1_data
# for STEP in 22000 17600 13200 8800 4400
# do
#         echo "7B_v1_data ($STEP) starting ..."
#         export BATCH_SIZE=32
#         export MODEL_PATH=/shared/csnell/data_study/7B_v1_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/7B_v1_data/$STEP/evals2
#         export CUDA_VISIBLE_DEVICES=0,4,5,6,7
#         mkdir $OUTPUT_PATH
#         # python main.py \
#         #         --model hf-causal-experimental \
#         #         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         #         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race,lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#         #         --num_fewshot=0 \
#         #         --device cuda \
#         #         --output_path $OUTPUT_PATH/0shot.json \
#         #         --batch_size $BATCH_SIZE \
#         #         --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hendrycksTest-*,triviaqa,bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "7B_v1_data ($STEP) done."
# done

# # 7B_v2_data
# for STEP in 22000 17600 13200 8800 4400
# do
#         echo "7B_v2_data ($STEP) starting ..."
#         export BATCH_SIZE=32
#         export MODEL_PATH=/shared/csnell/data_study/7B_v2_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/7B_v2_data/$STEP/evals2
#         export CUDA_VISIBLE_DEVICES=0,4,5,6,7
#         mkdir $OUTPUT_PATH
#         # python main.py \
#         #         --model hf-causal-experimental \
#         #         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         #         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race,lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#         #         --num_fewshot=0 \
#         #         --device cuda \
#         #         --output_path $OUTPUT_PATH/0shot.json \
#         #         --batch_size $BATCH_SIZE \
#         #         --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hendrycksTest-*,triviaqa,bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "7B_v2_data ($STEP) done."
# done

# conda activate torch_install

# # 1B_v1_data
# for STEP in 9800 5600 1400
# do
#         echo "1B_v1_data ($STEP) starting ..."
#         export BATCH_SIZE=256
#         export MODEL_PATH=/shared/csnell/data_study/1B_v1_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/1B_v1_data/$STEP/evals2_additional
#         export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         # python main.py \
#         #         --model hf-causal-experimental \
#         #         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         #         --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#         #         --num_fewshot=0 \
#         #         --device cuda \
#         #         --output_path $OUTPUT_PATH/0shot.json \
#         #         --batch_size $BATCH_SIZE \
#         #         --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "1B_v1_data ($STEP) done."
# done

# # 1B_v2_data
# for STEP in 14000 9800 5600 1400
# do
#         echo "1B_v2_data ($STEP) starting ..."
#         export BATCH_SIZE=256
#         export MODEL_PATH=/shared/csnell/data_study/1B_v2_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/1B_v2_data/$STEP/evals2_additional
#         export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         # python main.py \
#         #         --model hf-causal-experimental \
#         #         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         #         --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#         #         --num_fewshot=0 \
#         #         --device cuda \
#         #         --output_path $OUTPUT_PATH/0shot.json \
#         #         --batch_size $BATCH_SIZE \
#         #         --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "1B_v2_data ($STEP) done."
# done

# # 3B_v1_data
# for STEP in 44000 35200 28800 22400
# do
#         echo "3B_v1_data ($STEP) starting ..."
#         export BATCH_SIZE=128
#         export MODEL_PATH=/shared/csnell/data_study/3B_v1_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/3B_v1_data/$STEP/evals2_additional
#         export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         # python main.py \
#         #         --model hf-causal-experimental \
#         #         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         #         --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#         #         --num_fewshot=0 \
#         #         --device cuda \
#         #         --output_path $OUTPUT_PATH/0shot.json \
#         #         --batch_size $BATCH_SIZE \
#         #         --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "3B_v1_data ($STEP) done."
# done

# # 3B_v2_data
# for STEP in 44000 30800 17600 4400
# do
#         echo "3B_v2_data ($STEP) starting ..."
#         export BATCH_SIZE=128
#         export MODEL_PATH=/shared/csnell/data_study/3B_v2_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/3B_v2_data/$STEP/evals2_additional
#         export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         # python main.py \
#         #         --model hf-causal-experimental \
#         #         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         #         --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#         #         --num_fewshot=0 \
#         #         --device cuda \
#         #         --output_path $OUTPUT_PATH/0shot.json \
#         #         --batch_size $BATCH_SIZE \
#         #         --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "3B_v2_data ($STEP) done."
# done


# # log_loss_v_downstream_experiments

# conda activate torch_install

# # 13B_v1
# for STEP in 300000
# do
#         echo "13B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=24
#         export MODEL_PATH=/shared/csnell/openllama/13B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/13B_v1/$STEP/evals_2
#         export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         # python main.py \
#         #         --model hf-causal-experimental \
#         #         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         #         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#         #         --num_fewshot=0 \
#         #         --device cuda \
#         #         --output_path $OUTPUT_PATH/0shot.json \
#         #         --batch_size $BATCH_SIZE \
#         #         --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hendrycksTest-*,triviaqa' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "13B_v1 ($STEP) done."
# done
# for STEP in 400000 500000
# do
#         echo "13B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=24
#         export MODEL_PATH=/shared/csnell/openllama/13B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/13B_v1/$STEP/evals_2
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
#         echo "13B_v1 ($STEP) done."
# done

# 9/11/23

# data study additional evals

# conda activate torch_install

# # 1B_v1_data
# # for STEP in 14000 9800 5600 1400
# for STEP in 14000
# do
#         echo "1B_v1_data ($STEP) starting ..."
#         export BATCH_SIZE=8
#         export MODEL_PATH=/shared/csnell/data_study/1B_v1_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/1B_v1_data/$STEP/evals2_additional
#         export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         # python main.py \
#         #         --model hf-causal-experimental \
#         #         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         #         --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#         #         --num_fewshot=0 \
#         #         --device cuda \
#         #         --output_path $OUTPUT_PATH/0shot.json \
#         #         --batch_size $BATCH_SIZE \
#         #         --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "1B_v1_data ($STEP) done."
# done

# # 1B_v2_data
# for STEP in 14000 9800 5600 1400
# do
#         echo "1B_v2_data ($STEP) starting ..."
#         export BATCH_SIZE=8
#         export MODEL_PATH=/shared/csnell/data_study/1B_v2_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/1B_v2_data/$STEP/evals2_additional
#         export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         # python main.py \
#         #         --model hf-causal-experimental \
#         #         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         #         --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#         #         --num_fewshot=0 \
#         #         --device cuda \
#         #         --output_path $OUTPUT_PATH/0shot.json \
#         #         --batch_size $BATCH_SIZE \
#         #         --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "1B_v2_data ($STEP) done."
# done

# # 3B_v1_data
# for STEP in 44000 35200 28800 22400
# do
#         echo "3B_v1_data ($STEP) starting ..."
#         export BATCH_SIZE=4
#         export MODEL_PATH=/shared/csnell/data_study/3B_v1_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/3B_v1_data/$STEP/evals2_additional
#         export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         # python main.py \
#         #         --model hf-causal-experimental \
#         #         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         #         --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#         #         --num_fewshot=0 \
#         #         --device cuda \
#         #         --output_path $OUTPUT_PATH/0shot.json \
#         #         --batch_size $BATCH_SIZE \
#         #         --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "3B_v1_data ($STEP) done."
# done

# # 3B_v2_data
# for STEP in 44000 30800 17600 4400
# do
#         echo "3B_v2_data ($STEP) starting ..."
#         export BATCH_SIZE=4
#         export MODEL_PATH=/shared/csnell/data_study/3B_v2_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/3B_v2_data/$STEP/evals2_additional
#         export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         # python main.py \
#         #         --model hf-causal-experimental \
#         #         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         #         --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#         #         --num_fewshot=0 \
#         #         --device cuda \
#         #         --output_path $OUTPUT_PATH/0shot.json \
#         #         --batch_size $BATCH_SIZE \
#         #         --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "3B_v2_data ($STEP) done."
# done

# 7B_v1_data
# for STEP in 22000 17600 13200 8800 4400
# do
#         echo "7B_v1_data ($STEP) starting ..."
#         export BATCH_SIZE=4
#         export MODEL_PATH=/shared/csnell/data_study/7B_v1_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/7B_v1_data/$STEP/evals2
#         export CUDA_VISIBLE_DEVICES=1,2,5
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race,lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hendrycksTest-*,triviaqa,bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "7B_v1_data ($STEP) done."
# done

# 7B_v2_data
# for STEP in 22000 17600 13200 8800 4400
# do
#         echo "7B_v2_data ($STEP) starting ..."
#         export BATCH_SIZE=4
#         export MODEL_PATH=/shared/csnell/data_study/7B_v2_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/7B_v2_data/$STEP/evals2
#         export CUDA_VISIBLE_DEVICES=1,2,5
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race,lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hendrycksTest-*,triviaqa,bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "7B_v2_data ($STEP) done."
# done


# 9/8/23

# log_loss_v_downstream_experiments

# conda activate torch_install

# # 3B_v1
# for STEP in 10000 50000 100000 150000 200000 250000
# do
#         echo "3B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=8
#         export MODEL_PATH=/shared/csnell/openllama/3B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/3B_v1/$STEP/evals_2
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
#         echo "3B_v1 ($STEP) done."
# done

# # 7B_v1
# for STEP in 10000 50000 100000 150000 200000 250000
# do
#         echo "7B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=8
#         export MODEL_PATH=/shared/csnell/openllama/7B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/7B_v1/$STEP/evals_2
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
#         echo "7B_v1 ($STEP) done."
# done

# # 13B_v1
# for STEP in 20000 100000 200000 300000 400000 500000
# do
#         echo "13B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=4
#         export MODEL_PATH=/shared/csnell/openllama/13B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/13B_v1/$STEP/evals_2
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
#         echo "13B_v1 ($STEP) done."
# done




# 8/30/23

# # data study additional evals

# conda activate torch_install

# # 1B_v1_data
# for STEP in 14000 9800 5600 1400
# do
#         echo "1B_v1_data ($STEP) starting ..."
#         export BATCH_SIZE=8
#         export MODEL_PATH=/shared/csnell/data_study/1B_v1_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/1B_v1_data/$STEP/evals2_additional
#         export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola,coqa' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "1B_v1_data ($STEP) done."
# done

# # 1B_v2_data
# for STEP in 14000 9800 5600 1400
# do
#         echo "1B_v2_data ($STEP) starting ..."
#         export BATCH_SIZE=8
#         export MODEL_PATH=/shared/csnell/data_study/1B_v2_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/1B_v2_data/$STEP/evals2_additional
#         export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola,coqa' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "1B_v2_data ($STEP) done."
# done

# # 3B_v1_data
# for STEP in 44000 35200 28800 22400
# do
#         echo "3B_v1_data ($STEP) starting ..."
#         export BATCH_SIZE=4
#         export MODEL_PATH=/shared/csnell/data_study/3B_v1_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/3B_v1_data/$STEP/evals2_additional
#         export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola,coqa' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "3B_v1_data ($STEP) done."
# done

# # 3B_v2_data
# for STEP in 44000 30800 17600 4400
# do
#         echo "3B_v2_data ($STEP) starting ..."
#         export BATCH_SIZE=4
#         export MODEL_PATH=/shared/csnell/data_study/3B_v2_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/3B_v2_data/$STEP/evals2_additional
#         export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'lambada_openai,lambada_openai_cloze,copa,squad2,wikitext' \
#                 --num_fewshot=0 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/0shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'bigbench_mult_data_wrangling_*,bigbench_linguistic_mappings_*,bigbench_unit_conversion_*,bigbench_qa_wikidata,cola,coqa' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "3B_v2_data ($STEP) done."
# done


# 8/30/23

# log_loss_v_downstream_experiments

# conda activate torch_install

# 3B_v1
# for STEP in 10000 50000 100000 150000 200000 250000
# do
#         echo "3B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=8
#         export MODEL_PATH=/shared/csnell/openllama/3B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/3B_v1/$STEP/evals_1
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
#         echo "3B_v1 ($STEP) done."
# done

# 7B_v1
# for STEP in 10000 50000 100000 150000 200000 250000
# do
#         echo "7B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=8
#         export MODEL_PATH=/shared/csnell/openllama/7B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/7B_v1/$STEP/evals_1
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
#         echo "7B_v1 ($STEP) done."
# done

# 13B_v1
# for STEP in 300000
# do
#         echo "13B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=4
#         export MODEL_PATH=/shared/csnell/openllama/13B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/13B_v1/$STEP/evals_1
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
#         echo "13B_v1 ($STEP) done."
# done

# for STEP in 300000
# do
#         echo "13B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=4
#         export MODEL_PATH=/shared/csnell/openllama/13B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/13B_v1/$STEP/evals_1
#         export CUDA_VISIBLE_DEVICES=5,6,7
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
#         echo "13B_v1 ($STEP) done."
# done

# for STEP in 500000
# do
#         echo "13B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=4
#         export MODEL_PATH=/shared/csnell/openllama/13B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/13B_v1/$STEP/evals_1
#         export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
#         mkdir $OUTPUT_PATH
#         # python main.py \
#         #         --model hf-causal-experimental \
#         #         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#         #         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
#         #         --num_fewshot=0 \
#         #         --device cuda \
#         #         --output_path $OUTPUT_PATH/0shot.json \
#         #         --batch_size $BATCH_SIZE \
#         #         --no_cache
#         python main.py \
#                 --model hf-causal-experimental \
#                 --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
#                 --tasks 'hendrycksTest-*,triviaqa' \
#                 --num_fewshot=5 \
#                 --device cuda \
#                 --output_path $OUTPUT_PATH/5shot.json \
#                 --batch_size $BATCH_SIZE \
#                 --no_cache
#         echo "13B_v1 ($STEP) done."
# done

# for STEP in 10000 50000 100000 150000 200000 250000
# do
#         echo "13B_v1 ($STEP) starting ..."
#         export BATCH_SIZE=8
#         export MODEL_PATH=/shared/csnell/openllama/13B_v1/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/openllama/13B_v1/$STEP/evals_1
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
#         echo "13B_v1 ($STEP) done."
# done

# 8/25/23

# conda activate torch_install

# 1B_v1_data
# for STEP in 9800 5600 1400
# do
#         echo "1B_v1_data ($STEP) starting ..."
#         export BATCH_SIZE=8
#         export MODEL_PATH=/shared/csnell/data_study/1B_v1_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/1B_v1_data/$STEP/evals2
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
#         echo "1B_v1_data ($STEP) done."
# done

# # 1B_v2_data
# for STEP in 9800 5600 1400
# do
#         echo "1B_v2_data ($STEP) starting ..."
#         export BATCH_SIZE=8
#         export MODEL_PATH=/shared/csnell/data_study/1B_v2_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/1B_v2_data/$STEP/evals2
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
#         echo "1B_v2_data ($STEP) done."
# done

# 3B_v1_data
# for STEP in 35200 28800 22400
# do
#         echo "3B_v1_data ($STEP) starting ..."
#         export BATCH_SIZE=4
#         export MODEL_PATH=/shared/csnell/data_study/3B_v1_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/3B_v1_data/$STEP/evals2
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
#         echo "3B_v1_data ($STEP) done."
# done

# 3B_v2_data
# for STEP in 30800 17600 4400
# do
#         echo "3B_v2_data ($STEP) starting ..."
#         export BATCH_SIZE=4
#         export MODEL_PATH=/shared/csnell/data_study/3B_v2_data/$STEP/pytorch
#         export OUTPUT_PATH=/shared/csnell/data_study/3B_v2_data/$STEP/evals2
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
# # python main.py \
# #         --model hf-causal-experimental \
# #         --model_args pretrained=$MODEL_PATH,tokenizer=$MODEL_PATH,max_length=2048,use_accelerate=True \
# #         --tasks 'hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,boolq,rte,wic,record,anli_r1,anli_r2,anli_r3,truthfulqa_mc,race' \
# #         --num_fewshot=0 \
# #         --device cuda \
# #         --output_path $OUTPUT_PATH/0shot.json \
# #         --batch_size $BATCH_SIZE \
# #         --no_cache
# export BATCH_SIZE=8
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
# export BATCH_SIZE=8
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
