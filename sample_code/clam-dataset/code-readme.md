# Documentation of our sample code on CLaM dataset

### Description of the code
Model: ResNet18 model for breast cancer prediction of malignant and benign at the image-level on CLaM dataset. The case label is assigned to all images within a case for the training.

You can change the setting of the code in config_params (line 275 in main.py), e.g. you can change the ```device: 'cpu'``` if you don't have access to a cuda device. You can change the bit depth of the input image ```bitdepth: 12``` if you want to use 12 bit images instead of the 8 bit (current setup).

Note: This procedure of assigning groundtruth to the images does not assign correct groundtruth to each image. A case is labelled as malignant when atleast one of the images show signs of malignancy. Thus, all images within a malignant case may not be malignant. Thus, case-level training is more suited for CLaM dataset.

### Results
A successful execution of the sample code will show an AUC score of 0.6, F1 on positive (malignant) class is 0.4. 
