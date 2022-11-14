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
