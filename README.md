# Optical to infrared mapping of vapor-to-liquid phase change dynamics using generative machine learning

This repository contains the code for our paper:

**"Optical to infrared mapping of vapor-to-liquid phase change dynamics using generative machine learning"**  
*Siavash Khodakarami, Pouya Kabirzadeh, Chi Wang, Tarandeep Thukral Singh, Nenad Miljkovic*  

---
# Requirements

We suggest using Python 3.9+. The codes are tested with Pytorch 2.4.1 + Cuda 12.1 toolkit.

---
# Training
The following configuration is suggested for training the model with 3 gpus. For lower number of gpus, the gpu id should be changed (e.g., 0, 01, 012).    
!python train.py --name NAME  --direction AtoB --display_id -1 --input_nc 8 --output_nc 1 --batch_size 12 --n_epochs 40 --n_epochs_decay 90 --gan_mode='lsgan' --gpu_ids 012 --lr 0.00075 --netD 'basic' --netG 'unet_256' --n_layers_D 3 --lambda_L1 150 --lambda_D_cls 0.1 --lr_policy 'cosine'  

---
# Testing
We provide a template for testing the model after training and visualization in the [Inference notebook](https://github.com/SiaK4/cGAN-image-to-temperature-mapping/Inference.ipynb)  

--- 
# Dataset and Checkpoints
A sample of the dataset is provided in this repo. Full dataset will be available only upon reasonable requests from the corresponding authors of the paper.  
You can download the checkpoints for the generator and discriminator from [Link Text](https://huggingface.co/SiaK4/Image_to_IR/tree/main)
