#!/bin/bash
uname -a
#date
#env
date

DATASET=refcoco
DATA_PATH="/ailab_mat/dataset/refCOCO/images"
REFER_PATH="/ailab_mat/dataset/RIS"
BERT_PATH="/SSDe/heeseon_rho/src/CARIS/ckpt/bert-base-uncased/"
MODEL=caris
SWIN_TYPE=base
IMG_SIZE=448
ROOT_PATH="/SSDe/heeseon_rho/src/CARIS/output"
RESUME_PATH=${ROOT_PATH}/model_best_${DATASET}.pth
OUTPUT_PATH=${ROOT_PATH}/${DATASET}
SPLIT=val

# cd YOUR_CODE_PATH
python eval.py --model ${MODEL} --swin_type ${SWIN_TYPE} \
        --dataset ${DATASET} --split ${SPLIT} \
        --img_size ${IMG_SIZE} --resume ${RESUME_PATH} \
        --bert_tokenizer ${BERT_PATH} --ck_bert ${BERT_PATH} \
        --refer_data_root ${DATA_PATH} --refer_root ${REFER_PATH} 2>&1 | tee ${OUTPUT_PATH}/eval-${SPLIT}.txt
