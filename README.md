## 1. Introduction
Code used for publication in:
Tee, Y.H., Goh, W.J., Yong, X. et al. Actin polymerisation and crosslinking drive left-right asymmetry in single cell and cell collectives. Nat Commun 14, 776 (2023). https://doi.org/10.1038/s41467-023-35918-1


Input images: 512 x 512 pixel images of single cells patterned on 1800µm<sup>2</sup> micro-patterns. 

Code has been tested on Windows 10 and CentOS 7 (Linux)

## 2. Installation
[Fiji / ImageJ](https://imagej.net/Fiji/Downloads)

[MiniConda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

[MATLAB R2020a](https://www.mathworks.com/products/new_products/release2020a.html)

[CUDA](https://developer.nvidia.com/cuda-10.1-download-archive-base/) version 10.1 (Optional)


## 3. Running Image Analysis
### 3.1. Pre-processing
Fiji Scripts for pre-processing have been included for convenience. Alternatively, save image stacks as single 8-bit images in a folder labelled 'actin' (fixed data) or '<4-digit-index>/actin' (live data).

**Live Images**

Splits live movies for each x-y position into a folder (eg. '0001/actin', '0002/actin', ...).

**Fixed Images**

Splits image stacks into folder labelled 'actin'. Converts images to 8-bit and enhance brightness/contrast. For the included test images, the root directory should be "Test_data".

###  3.2. Actin Segmentation
Segmentation of radial fibres using ResNet-50

Install dependencies:
```
cd Actin_segmentation  # Path to Actin_segmentation folder
conda create -n new_environment python=3.6  -y  
conda activate new_environment
pip install -r requirements.txt
conda install tensorflow=1.14.0 ipykernel --y
python -m ipykernel install --user  --name new_environment --display-name "Cell-Chirality-Analysis"

```

Open and run actin_predict.ipynb using Jupyter Lab or Jupyter Notebook. 
```bash
jupyter lab
```
Proceed to select the kernel "Cell-Chirality-Analysis"
Note: ensure that the pre-process steps have been completed. 

###  3.3. MATLAB analysis
By default, the script run_analysis.m will skip folders that have already been processed. This behaviour can be changed by changing the 'skip_completed' variable to false. 

To run the script, first change the paths in ```paths.txt```. 
For example: 
```MATLAB
%  Enter paths for MATLAB analysis here, separated by newline

../Test_data
``` 
Open matlab and execute ```run_all.m``` in terminal.
The following example was run using a bash terminal. 
```bash
-bash-4.2$ matlab
MATLAB is selecting SOFTWARE OPENGL rendering.

                < M A T L A B (R) >
        Copyright 1984-2020 The MathWorks, Inc.
    R2020a Update 5 (9.8.0.1451342) 64-bit (glnxa64)
                August 6, 2020

 
To get started, type doc.
For product information, visit www.mathworks.com.
 
>> cd Matlab_analysis

matlab -r run_all
```
This script works on MATLAB versions after 2018. 

To check for completion of MATLAB analysis, check that ```matlab_log.txt``` in each of the data folders contains the following: 
```
0-2 completed
2-4 completed
4-6 completed
6-8 completed
8-10 completed
10-12 completed
12-14 completed
14-16 completed
```

###  3.4. Post-processing and data extraction
Change the variable ```rootdir``` in Cell 2, and the name of the output folder ```output_folder```. This notebook will iteratively search for, and analyse folders on which the MATLAB analysis has been successfully completed. All data can be found in ```output_folder```.
