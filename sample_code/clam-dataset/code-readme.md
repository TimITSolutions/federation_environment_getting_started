Our sample code for breast cancer prediction of malignant and benign at the image-level on CLaM dataset. The case label is assigned to all images within a case for the training. 
Input: all images in CLaM dataset
Output: malignant and benign
Model used: ResNet18.

You can change the setting of the code in config_params (line 275 in main.py), e.g. you can change the ```device: 'cpu'``` if you don't have access to a cuda device. You can change the bit depth of the input image ```bitdepth: 12``` if you want to use 12 bit images instead of the 8 bit (current setup).
