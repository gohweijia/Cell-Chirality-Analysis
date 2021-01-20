## 1. Introduction
Code used for publication

Input images: 512 x 512 pixels images of single cells patterned on 1800µm<sup>2</sup> micro-patterns. 

Code has been tested on MacOS and Windows 10

## 2. Installation
[Fiji / ImageJ](https://imagej.net/Fiji/Downloads)

[MiniConda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

[MATLAB R2020a](https://www.mathworks.com/products/new_products/release2020a.html)

[Test dataset](http://www.google.com/)  ***TODO***

## 3. Running Image Analysis
### 3.1. Pre-processing
Fiji Scripts for pre-processing have been included for convenience. Alternatively, save image stacks as single 8-bit images in a folder labelled 'actin' (fixed data) or '<4-digit-index>/actin' (live data).

**Live Images**

Splits live movies for each x-y position into a folder (eg. '0001/actin', '0002/actin', ...).

**Fixed Images**

Splits image stacks into folder labelled 'actin'. Converts images to 8-bit and enhance brightness/contrast.

###  3.2. Actin Segmentation
Segmentation of radial fibres using ResNet-50

Install dependencies:
```
cd Actin_segmentation  # Path is just an example
conda create -n new_environment python=3.6  -y
conda activate new_environment
pip install -r requirements.txt
conda install tensorflow=1.14.0 --y
```

Open actin_predict.ipynb using Jupyter Lab or Jupyter Notebook. 
```bash
jupyter lab
```
Note: ensure that the pre-process steps have been completed. 
###  3.3. MATLAB analysis
Code written by Ong Hui Ting. 


###  3.4. Post-processing and data extraction
