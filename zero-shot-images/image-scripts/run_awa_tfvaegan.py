#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:52:45 2019
@author: akshita
"""
import os

# continous attribute

# os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python train_tfvaegan_inductive.py --gammaD 10 \
# --gammaG 10 --gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 \
# --class_embedding att --nepoch 120 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --nclass_all 50 --dataroot data --dataset AWA2 \
# --batch_size 64 --nz 85 --latent_size 85 --attSize 85 --resSize 2048 \
# --lr 0.00001 --classifier_lr 0.001 --recons_weight 0.1 --freeze_dec \
# --feed_lr 0.0001 --dec_lr 0.0001 --feedback_loop 2 --a1 0.01 --a2 0.01''')

# binary attribute

# os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python train_tfvaegan_inductive.py --gammaD 10 \
# --gammaG 10 --gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 \
# --class_embedding binaryAtt --nepoch 120 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --nclass_all 50 --dataroot data --dataset AWA2 \
# --batch_size 64 --nz 85 --latent_size 85 --attSize 85 --resSize 2048 \
# --lr 0.00001 --classifier_lr 0.001 --recons_weight 0.1 --freeze_dec \
# --feed_lr 0.0001 --dec_lr 0.0001 --feedback_loop 2 --a1 0.01 --a2 0.01''')

# bit_encoding

# os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python train_tfvaegan_inductive.py --gammaD 10 \
# --awa_encoding bit_encoding --gammaG 10 --gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 \
# --class_embedding att --nepoch 120 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --nclass_all 50 --dataroot data --dataset AWA2 \
# --batch_size 64 --nz 85 --latent_size 85 --attSize 85 --resSize 2048 \
# --lr 0.00001 --classifier_lr 0.001 --recons_weight 0.1 --freeze_dec \
# --feed_lr 0.0001 --dec_lr 0.0001 --feedback_loop 2 --a1 0.01 --a2 0.01''')

# label_encoding
os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python train_tfvaegan_inductive.py --gammaD 10 \
--awa_encoding label_encoding --gammaG 10 --gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 \
--class_embedding att --nepoch 120 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--nclass_all 50 --dataroot data --dataset AWA2 \
--batch_size 64 --nz 85 --latent_size 85 --attSize 85 --resSize 2048 \
--lr 0.00001 --classifier_lr 0.001 --recons_weight 0.1 --freeze_dec \
--feed_lr 0.0001 --dec_lr 0.0001 --feedback_loop 2 --a1 0.01 --a2 0.01''')

# one_hot_encoding
# os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python train_tfvaegan_inductive.py --gammaD 10 \
# --awa_encoding one_hot_encoding --gammaG 10 --gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 \
# --class_embedding att --nepoch 120 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --nclass_all 50 --dataroot data --dataset AWA2 \
# --batch_size 64 --nz 85 --latent_size 85 --attSize 85 --resSize 2048 \
# --lr 0.00001 --classifier_lr 0.001 --recons_weight 0.1 --freeze_dec \
# --feed_lr 0.0001 --dec_lr 0.0001 --feedback_loop 2 --a1 0.01 --a2 0.01''')