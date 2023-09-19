# Exploring 3D reconstruction from RGB images of partially occluded objects
This project aims to explore the impact of occlusion on [Pix2Vox](https://arxiv.org/abs/1901.11153), a multi-view reconstruction model, and possible approaches for improving its generated 3D shapes. In this work erasing data augmentation techniques have been used for generating occluded versions of the [ShapeNet](https://arxiv.org/abs/1512.03012) dataset and [StableDiffusion](https://arxiv.org/abs/2112.10752) for inpainting the occluded images before feeding them to Pix2Vox. Results show that the developed approach improves the performance of the baseline by 51\% on severe occlusion. Interestingly, fine-tuning the Pix2Vox model using occluded images by 30% to 40%, further improves the performance by 97% on severe occlusion. Additionally, the approach improves the generalisation and performance on the [Pix3D](https://arxiv.org/abs/1804.04610) dataset by 16%.

### 3D reconstruction sample results
![3D-voxels-samples](https://github.com/DavideGrecoGit/3D-reconstruction-under-occlusion/blob/main/CompareFineTunedModels/3D_voxels_samples.png?raw=true)

### Inpainting sample results
![inpainting-samples](https://github.com/DavideGrecoGit/3D-reconstruction-under-occlusion/blob/main/DatasetGeneration/Inpainting_samples.png)


# Downloads
| Fine-tuned model | Link | Size |
| ------ | ------ | ------ |
| Erasing_10_20 | https://drive.google.com/file/d/1uY-8hL4IyEawBANTp96bHdRS2mcVQPwX/view?usp=drive_link | 1.12 GB |
| Erasing_20_30 | https://drive.google.com/file/d/1kchDamWWg1qXQCJPJtkqT2R3wP37J3g8/view?usp=drive_link | 1.11 GB |
| Erasing_30_40 | https://drive.google.com/file/d/1OfWwCTJDL7DgFDYk_3zIVzddPkt8gEMG/view?usp=drive_link | 1.12 GB |
| Erasing_10_40 | https://drive.google.com/file/d/1mndFOrewWJJJCx23ZzPd-fkQUjRN2T-8/view?usp=drive_link | 1.12 GB |

| Dataset | Link | Size |
| ------ | ------ | ------ |
| ORE_Random_(Occluded/Inpainted/Combined)_(15/25/50) |https://drive.google.com/file/d/1vIdCUMJMNSn2Sb-HlorfCY0emoMWA_kJ/view?usp=drive_link | 660.7 GB |
| ShapeNet rendering images | http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz | 11.5 GB |
| ShapeNet voxelized models | http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz | 21.2 MB |
| Pix3D images & voxelized models | http://pix3d.csail.mit.edu/data/pix3d.zip | 3.5 GB |


# File structure
### Pix2Vox
Instructions on how to run Pix2Vox can be found [here](https://github.com/hzxie/Pix2Vox). 
The code in this repository is a modified version that supports erasing for augmenting input images and allows to save the voxels of the specified samples as images.

Additional files are:
- **FineTuning**
    - _Purpose_: fine-tuning of a given checkpoint 
    - *What to change*
        - out_path -> output path for the generated checkpoints, train and validation files (.csv), and logs of Tensorboard
        - taxonomy -> path to the ShapeNet taxonomy file
        - weights -> path to the model to fine-tune
        - Fine-tuning hyperparameters
- **Evaluation**
    - _Purpose_: test a given list of models on a given list of datasets
    - *What to change*
        - taxonomy -> path to the ShapeNet taxonomy file
        - rendering_paths -> list of paths to the evaluation datasets
        - ckpts -> list of paths to the models to evaluated
        - out_path -> output path for the generated evaluation files (.csv), voxel and gt images of the selected samples, and logs of Tensorboard

- **CompareFineTuned**
    - _Purpose_: compare and plot the impact of fine-tuning hyperparameters and their evaluation results
    - *What to change*
        - pandas_path -> path to the train and validation csv files generated by "FineTuning". Or path to the evaluation csv files generated by "Evaluation"


### Generating inpainting datasets
The code and Instructions for runnng StableDiffusion as an API is provided [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

The Stable diffusion checkpoint used for inpainting can be downloaded from [here](https://huggingface.co/runwayml/stable-diffusion-inpainting).

After having the StableDiffusion running as an API in the background, the "Inpainting" file can be used to inpaint a specified dataset.

- **Inpainting**
    - *Purpose*: inpaint an occluded variant of ShapeNet or to combine the original dataset with the inpainted version
    - *What to change*
        - taxonomy_file ->  path to the ShapeNet taxonomy file
        - output_path -> path on which save the generated 
        - occluded_path -> path to the dataset to inpaint
        - inpainting_path -> path of the newly inpainted dataset
        - url -> API url of StableDiffusion


### Occluded ShapeNet and Pix3D datasets
Following the explanation of the files used for the generation and analysis of the occluded variants of ShapeNet and the pre-processing of Pix3D.

- **OccludeShapeNet-Chair**
    - *Purpose*: generate the occluded variations of the ShapeNet dataset
    - *What to change*
        - taxonomy_file -> path to the ShapeNet taxonomy file
        - renders_path -> path to the original ShapeNet renders
        - output_path -> path on which save the generated dataset
- **Pix3D_preprocessing**
    - *Purpose*: applying the given mask to the input images, convert the obj models to binvox.
    - *What to change*
        - pix3d_path -> path to the Pix3D dataset
        - chair_dir -> path to the Pix3D model chair folder
        - Need to download the [mesh voxelizer](https://www.patrickmin.com/binvox/) and place in the same directory of this file
- **AnalyseDatasets**
    - *Purpose*: Calculate and plot the information loss of an occluded variant of ShapeNet
    - *What to change*
        - datasets_path -> path to the occluded dataset
        - gt_path -> path to the original dataset
        - save_path ->  output path for the generated .csv files 

### BibTex

(Just in case someone needs it! :D)

```
@misc{Greco2023exploring,
  title = {Exploring 3D reconstruction from RGB images of partially occluded objects},
  author = {Davide Greco},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DavideGrecoGit/3D-reconstruction-under-occlusion}},
  commit = {c9f3d6f}
}
```

