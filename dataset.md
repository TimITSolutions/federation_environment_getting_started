# CLaM Dataset
<ins>C</ins>ase-<ins>La</ins>belled <ins>M</ins>ammography (CLaM) dataset contains mammography exams from Ziekenhuis Groep Twente (ZGT), The Netherlands, taken between 2013 to 2020. Our complete CLaM dataset is stored at ZGT and is not downloadable or directly accessible, but can be used for training AI models through our [platform](https://fe.zgt.nl). Details of the dataset can be found in our paper (in progress). <br/> 
The structure of the dataset is as follows:
```
dataset
| 
+--CLaM
|  |
|  +--S1-A1
|  |  |
|  |  +--LCC
|  |  |  |
|  |  |  \--1.png
|  |  |  \--1.npy
|  |  +--LMLO
|  |  |  |
|  |  |  |
+--clam-details-case.csv
+--clam-details-image.csv 
```

## Example data
We provide a sample of the CLaM dataset, [CLaM-sample](./dataset) in this repository for users to prepare their code that can work on the CLaM dataset. CLaM-sample contains 10 cases or mammography exams (S01-A01, S02-A02, ..) from 10 deceased patients (P01, P02, ...). A mammography exam/case contains images of standard views from left (L) and right (R) breast - LMLO, LCC, RMLO and RCC and can also contain additional views - LXCCL, RXCCL, LLM, RLM, LML, RML. Each image folder, e.g. S01-A01/LMLO contains 2 images - 1.png (image in 8 bit) and 1.npy (image in 12 bit). 

The CLaM dataset stored in our [platform](https://fe.zgt.nl) reflects a similar structure and can be accessed similarly.

## csv files
[clam-details-case.csv](./dataset/clam-details-case.csv): list of the cases in the dataset and their corresponding case-level diagnosis of malignant and benign. The columns in the csv file are: 
|Column name      | Description                                                                             |
|-----------------|-----------------------------------------------------------------------------------------|
|Patient_Id       | Unique id of the patient to whom the exam belongs.                                      |
|CaseName         | Name of a case or exam, e.g. S1-A1. Unique for each case.                               |
|CasePath         | Path to the case, e.g. ./dataset/CLaM/S1-A1, then CasePath is CLaM/S1-A1                |
|Study_Description| Description of the exam                                                                 |
|Views_4          | Only the standard views are mentioned                                                   | 
|Views            | mentions all views available for the case, including both standard and additional views |
|Groundtruth      | Case-level or exam-level diagnosis of malignant or benign                               |
|Split            | indicates whether the case is included in the train or test set.                        |


[clam-details-image.csv](./dataset/clam-details-image.csv): list of images in the cases. The columns in the csv file are as follows:<br/>
|Column name      | Description                                                                                                         |
|-----------------|---------------------------------------------------------------------------------------------------------------------|
|Patient_Id       | Unique id of the patient to whom the exam belongs, e.g. P1, P2..                                                    |
|CaseName         | Name of a case or exam, e.g. S1-A1. Unique for each case                                                           |
|CasePath         | Path to the case, e.g. ./dataset/CLaM/S1-A1, then CasePath is CLaM/S1-A1                                            |
|ImagePath        | Path to each image in a case, e.g. for ./dataset/CLaM/S1-A1/LCC/1.png, the ImagePath is CLaM-sample/S1-A1/LCC/1.png |
|Study_Description| Description of the exam                                                                                             |
|Views            | view of the image, e.g. LCC, LMLO...                                                                                |
|CaseLabel        | Case-level diagnosis of malignant or benign assigned to each image in a case                                        |
|Split            | indicates whether the image is included in the train or test set. A case assigned to the train/test set has all its images assigned to the same set.|

## How to access CLaM on the platform
The CLaM dataset on the [platform](https://fe.zgt.nl) is located under ```/mnt/dataset```. Each case can be accessed using the path ```/mnt/dataset``` + column name ```CasePath``` in ```/mnt/dataset/clam-details-case.csv``` and each image can be accessed using the path ```/mnt/dataset``` + column name ```ImagePath``` in ```/mnt/dataset/clam-details-image.csv```.
