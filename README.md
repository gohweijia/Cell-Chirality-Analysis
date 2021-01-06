# Cell-Chirality-Analysis
Code used for publication

Input images: 512 x 512 pixels images of single cells patterned on 1800µm<sup>2</sup> micro-patterns. 

- [Cell-Chirality-Analysis](#cell-chirality-analysis)
  - [Pre-processing](#pre-processing)
  - [Actin Segmentation](#actin-segmentation)
  - [MATLAB analysis](#matlab-analysis)
  - [Post-processing and data extraction](#post-processing-and-data-extraction)

## Pre-processing
Fiji Scripts for pre-processing have been included for convenience. Alternatively, save image stacks as image sequence using Fiji in a folder labelled 'actin' (fixed) or '<4-digit-index>/actin' (live).

**Live Images**

Splits live movies for each x-y position into a folder (eg. '0001/actin', '0002/actin', ...).

**Fixed Images**

Splits image stacks into folder labelled 'actin'. Converts images to 8-bit and enhance brightness/contrast.

##  Actin Segmentation
Segmentation of radial fibres using ResNet-50
Install dependencies:
```
cd Actin_segmentation
pip install -r requirements.txt
```

##  MATLAB analysis
Code written by Ong Hui Ting. 


##  Post-processing and data extraction
