# Timelog

- Reconstructing 3D Object Models from RGB Image/Video Data
- Davide Greco
- 2402884G
- Shu-lim Edmond Ho

## Week 1

### 20 Sept 2022

- _3 hours_ Read the project guidance notes

## Week 2

### 28 Sept 2022

- _0.5 hour_ Read 3D car shape reconstruction from a contour sketch using GAN and lazy learning paper
- _1.5 hour_ Read Pix2Vox paper
- _1 hour_ Trying to train Pix2Vox using Colab (Not successful)

### 30 Sept 2022

- _1 hour_ Read 3D-R2N2 paper
- _0.5 hour_ Setting up GitLab repository
- _0.5 hour_ Trying to use pre-trained Pix2Vox models using my Laptop (Not successful)
- _0.5 hour_ Meeting with supervisor
- _0.5 hour_ Updating GitLab repository (Creating Issues, First milestone, Labels and writing the meeting's log)
- _1 hour_ Searching for papers that tackle the problem from different views. Found a nice review covering 149 different papers: Han, X., Laga, H. and Bennamoun, M., 2021. Image-Based 3D Object Reconstruction: State-of-the-Art and Trends in the Deep Learning Era. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(5), pp.1578-1604.

## Week 3

### 3 Oct 2022

- _3.5 hours_ Read The paper "Image-Based 3D Object Reconstruction: State-of-the-Art and Trends in the Deep Learning Era."
- _2 hours_ Trying to use pre-trained Pix2Vox models on Colab (Not possible due to lack of space)
- _0.5 hour_ Read about domain randomization
- _1.5 hour_ Read Occupancy Networks: Learning 3D Reconstruction in Function Space

### 4 Oct 2022

- _1.5 hour_ Meeting preparation (writing down questions, possible ideas to pursue and salient information I found reading papers)
- _1.5 hour_ Exploring the literature looking for 3D object generation using GANs (SDF-StyleGAN seems promising)
- _1 hour_ Exploring the literature looking for 2D image generation using Gans (StyleGAN-XL seems promising)

### 5 Oct 2022

- _1 hour_ Making notes of "Image-Based 3D Object Reconstruction: State-of-the-Art and Trends in the Deep Learning"
- _1 hour_ Updating GitLab repository (Creating Issues, updating wiki content and sidebar, updating the plan)
- _1.5 hour_ Successfully running Stable Diffusion on Colab

### 6 Oct 2022
- _0.5 hour_ Seaching how ONet deals with background in real-world images: the background is removed using masks before giving it to the model.

### 7 Oct 2022
- _0.5 hour_ Setting up script for committing from Colab to GitLab

### 9 Oct 2022
- _3.5 hours_ Exploring Img2Img and Dreambooth for 2D images generation
- _1 hour_ Running pre-trained Pix2Vox models 
- _1 hour_ Trying to use ONet on Colab (pre-processing the input is not feasible due to hardware requirements)

## Week 4 
### 10-16 Oct 2022 - No updates due to Covid

## Week 5
### 17-22 Oct 2022 - No updates due to Covid
### 23 Oct 2022 
- _1 hour_ Setting up conda environment for 3DAttriFlow
- _1 hour_ Updating GitLab repository

## Week 6
### 24 Oct 2022
- _3 hours_ Finishing setting up 3DAttriFlow and Successfully running it
- _1.5 hours_ Researching about inpainting with Python

### 25 Oct 2022
- _0.5 hour_ Reading about Mask R-CNN
- _1 hours_ Reading about Pallette and MAT
- _2 hours_ Successfully running MAT
- _0.5 hour_ Reading about ImageNet dataset

### 28 Oct 2022
- _2 hours_ Resizing and pre-processing of ShapeNet dataset to be used with MAT
- _1.5 hours_ Cuda and torch installation (Not successful)

### 29 Oct 2022
- _3 hours_ Trying to install torch compatible with my CUDA and the requirements of MAT
- _2 hours_ Trying to solve issues with MAT and its libraries 
- _2 hours_ Installing Lambda Stack

## Week 7
### 31 Oct 2022
- _1 hour_ Setting up "test" training of 3DAttriFlow (Not successful)

### 1 Nov 2022
- _2 hours_ Debugging 3DAttriFlow default code for training
- _1 hour_ Debugging additional errors of 3DAttriFlow's default code
- _1 hour_ Setting up Palette's config and downloading test dataset, trying to test it but got errors: the given test dataset and .list file do not match 

## Week 8
### 7 Nov 2022
- _1 hour_ Eploring Pixel2Mesh
- _1.5 hours_ Exploring conditional image generation
- _1 hour_ Successfully training Palette on subset of Places365

### 8 Nov 2022
- _1 hour_ Configuring Palette to be trained on a subset of ShapeNet
- _1.5 hours_ Experimenting with different files and configurations
- _1 hour_ Updating and Organising GitLab repository

### 13 Nov 2022
- _2 hours_ Trying to remotely access Desktop PC (outside LAN) using Remmina (With and without SSH) and NoMachine

## Week 9
### 14 Nov 2022
- _1 hour_ Installing dependencies for OccupancyNetworks
- _1 hour_ Trainig Palette using irregular masks and ShapeNet subset
- _1.5 hours_ Trying to fix corrupted hard disk where the data for OccupancyNetworks is stored (data has been lost)
- _1 hour_ Setting up and launching training of Pix2Vox
- _0.5 hour_ Updating Wiki with Palette training results
- _1 hour_ Fixing issues with Pix2Vox and starting training on full Chair subset 
- _1.5 hour_ Trying different batch size and num_workers for Palette

### 15 Nov 2022
- _1 hour_ Trying to fix corrupted hard disk
- _0.5 hour_ Masking Pix3D images using the provided masks
- _1 hour_ Exploring state of the art for object detection (Considreing mmdetection)
- _2 hours_ Making diagrams
- _0.5 hour_ Updating the wiki

### 16 Nov 2022
- _1 hour_ Setting up Palette for training without a starting checkpoint
- _1 hour_ Clearing log data and making graphs of Palette training
- _0.5 hour_ Setting up Palette for training using place2 pre-trained checkpoint

### 17 Nov 2022
- _0.5 hour_ Clearing log data and making graphs of Palette training
- _1.5 hours_ Setting up Palette on Colab
- _0.5 hour_  Making some tries and launching the training of Palette on Colab

### 23 Nov 2022
- _0.5 hour_ Writing a script to convert ShapeNet (chair category) to gray
- _0.5 hour_ Setting up Palette config for training

### 5 Dec 2022
- _1 hour_ Augmenting data and palette training

### 12 Dec 2022
- _3.5 hours_ Defining the plan of the next semester

### 13 Dec 2022
- _3 hours_ Exploring 3D diffusion models such as Get3D
- _3 hours_ Exploring image segmentation models and using Mask R-CNN TF2 model 

### 17 Dec 2022
- _3 hours_ Fixing issues with Pix2Vox DataLoader

### 18 Dec 2022
- _1.5 hours_ Setting up and starting training of Pix2Vox with Random Erasing, masks between 32 and 64

### 19 Dec 2022
- _1.5 hours_ Setting up training of Pix2Vox with Random Erasing, masks between 64 and 128

### 20 Dec 2022
- _1.5 hours_ Setting up training of Pix2Vox with Random Erasing, masks between 32 and 128

### 28 Dec 2022
- _4 hours_ Reading about GANs and fine-tuning FasterRCNN for single-class detection

### 4 Jan 2023
- _1 hour_ Fixing 3D plot visualisation
- _1 hour_ Fixing voxel rotation
- _2 hour_ Testing and exploring evaluation methods

### 5 Jan 2023
- _4 hours_ Generating OccShapeNet for testing
- _2.5 hours_ Writing evaluation script

### 6 Jan 2023
- _5 hours_ Continuing evaluation script
- _1 hour_ Running evaluations and collecting results

### 7 Jan 2023
- _3 hours_ Testing SD

### 9 Jan 2023
- _1 hour_ Trying to evaluate Pix2Vox on Pix3D
- _2 hours_ Solving issues and converting Pix3D obj models to binvox

