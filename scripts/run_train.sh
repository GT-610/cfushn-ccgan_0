#!/bin/bash

#ROOT_PATH="D:/local_wd/tmp/CcGAN_tutorial/UTKFace/CcGAN"
#DATA_PATH="C:/Users/DX/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/UTKFace"
#EVAL_PATH="D:/local_wd/tmp/CcGAN_tutorial/UTKFace/evaluation"
#niqe_dump_path="D:/local_wd/tmp/CcGAN_tutorial/CcGAN_TPAMI_NIQE/UTKFace/NIQE_64x64/fake_data"
ROOT_PATH="/Volumes/gong/blood_files/---AI---/workdir/cfushn-ccgan_0"
DATA_PATH="/Volumes/gong/blood_files/---AI---/workdir/cfushn-ccgan_0/data"
EVAL_PATH="/Volumes/gong/blood_files/---AI---/workdir/cfushn-ccgan_0/evaluation"
niqe_dump_path="/Volumes/gong/blood_files/---AI---/workdir/cfushn-ccgan_0/NIQE_64x64/fake_data"


SEED=2021
NUM_WORKERS=0
MIN_LABEL=1
MAX_LABEL=60
IMG_SIZE=64
MAX_N_IMG_PER_LABEL=99999
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=200

BATCH_SIZE_G=256
BATCH_SIZE_D=256
NUM_D_STEPS=2
SIGMA=-1.0
KAPPA=-1.0
LR_G=1e-4
LR_D=1e-4
NUM_ACC_D=1
NUM_ACC_G=1

GAN_ARCH=SNGAN
LOSS_TYPE=vanilla

DIM_GAN=256
DIM_EMBED=128

NITERS=10
resume_niter=0
#resume_niter=40000
## 设置为0,则从头开始训练；设置为其他值，则载入相应的checkpoint（若有），然后继续训练。

python ../train.py \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --eval_ckpt_path $EVAL_PATH \
    --seed $SEED \
    --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL \
    --max_label $MAX_LABEL \
    --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL \
    --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN_arch $GAN_ARCH \
    --niters_gan $NITERS \
    --resume_niters_gan $resume_niter \
    --loss_type_gan $LOSS_TYPE \
    --num_D_steps $NUM_D_STEPS \
    --save_niters_freq 5000 \
    --visualize_freq 1000 \
    --batch_size_disc $BATCH_SIZE_D \
    --batch_size_gene $BATCH_SIZE_G \
    --num_grad_acc_d $NUM_ACC_D \
    --num_grad_acc_g $NUM_ACC_G \
    --lr_g $LR_G \
    --lr_d $LR_D \
    --dim_gan $DIM_GAN \
    --dim_embed $DIM_EMBED \
    --kernel_sigma $SIGMA \
    --threshold_type soft \
    --kappa $KAPPA \
    --gan_DiffAugment \
    --gan_DiffAugment_policy color,translation,cutout \
    --comp_FID

    # --dump_fake_for_NIQE --niqe_dump_path $niqe_dump_path