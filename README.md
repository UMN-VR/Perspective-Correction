# Perspective-Correction-Python
Automated Perspective Correction using deep neural networks transformation matrixes, OpenCV and Numpy

# root_painter

This python script relies on segmentations created by root painter to frame the water pouch and create a consistent image even with changing camera positions in diferent dates. 


## An Overview of the data processing pipeline:

### Step 1: Create deep neural network activation mask

### Step 2: Find 4-sided polygon defining the outline of the water pouch

### Step 3: Calculate transformation matrix

### Step 4: Transform image using matrix to obtain corrected version
