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
+--clam-details-case-extrainfo.csv
+--clam-details-image.csv 
```

## Example data
We provide a sample of the CLaM dataset, [CLaM-sample](./dataset) in this repository for users to prepare their code that can work on the CLaM dataset. CLaM-sample contains 10 mammography exams or cases (S1-A1, S2-A2, ..) from 10 patients (P1, P2, ...). A mammography exam/case contains images of standard views, namely, mediolateral oblique (MLO) and craniocaudal (CC) from left (L) and right (R) breast, resulting in LMLO, LCC, RMLO and RCC images. The mammography exams can also contain additional views, namely, exaggerated craniocaudal view (XCCL), lateromedial (LM), mediolateral (ML), resulting in LXCCL, RXCCL, LLM, RLM, LML and RML images. 

Each image folder, e.g. S01-A01/LMLO contains 2 images - 1.png (image in 8 bit) and 1.npy (image in 12 bit). 

The CLaM dataset stored in our [platform](https://fe.zgt.nl) reflects a similar structure and can be accessed similarly.

## csv files
[clam-details-case-extrainfo.csv](./dataset/clam-details-case-extrainfo.csv): list of the cases in the dataset and their corresponding case-level diagnosis of malignant and benign. The columns in the csv file are: 
|Column name                    | Description                                                                                      | Value |
|-------------------------------|--------------------------------------------------------------------------------------------------|-------|
|Patient_Id                     | Unique id of the patient to whom the exam belongs.                                  | P1, P2, .. ; datatype: string|
|CaseName                       | Name of a case or exam. Unique for each case                                  | S1-A1, S2-A2, .. ; datatype: string |
|CasePath                       | Path to the case                       | for ./dataset/CLaM/S1-A1, the CasePath is CLaM/S1-A1; datatype: string |
|Study_Description              | Description of the exam                                                                          | datatype: string |
|Views_4                        | All standard views in the case are mentioned.                                                    | A case containing LCC, LMLO, RCC and LXCCL would have a value LCC+LMLO+RCC in this column; datatype: string | 
|Views                          | mentions all views available for the case, including both standard and additional views.          | A case containing LCC, LMLO, RCC and LXCCL would have LCC+LMLO+RCC+LXCCL in this column; datatype: string |
|BIRADS_combined_casebased      | BIRADS score from the mammography exam. | 0, 1, 2, 3, 4, 4a, 4b, 4c, 5, 6, NULL; datatype: string |
|BIRADS_combined_pathwaybased   | BIRADS score from the complete diagnostic pathway of the patient.  | 0, 1, 2, 3, 4, 4a, 4b, 4c, 5, 6, NULL; datatype: string|
|BreastDensity_standardized     | Breast density of the patient.                                        | A, B, C, D, NULL; data type: string |
|Age                            | Age of the patient.                                                                               | datatype: float |
|Groundtruth                    | Case-level or exam-level diagnosis of malignant or benign.          | malignant, benign; datatype: string|
|Split                          | indicates whether the case is included in the train or test set.            | train, test; datatype: string|


[clam-details-image.csv](./dataset/clam-details-image.csv): list of images in the cases. The columns in the csv file are as follows:<br/>
|Column name      | Description                                                                                                         |
|-----------------|---------------------------------------------------------------------------------------------------------------------|
|Patient_Id       | Unique id of the patient to whom the exam belongs, e.g. P1, P2.                                                    |
|CaseName         | Name of a case or exam, e.g. S1-A1, S2-A2. Unique for each case.                                                            |
|CasePath         | Path to the case, e.g. for ./dataset/CLaM/S1-A1, the CasePath is CLaM/S1-A1                                         |
|ImagePath        | Path to each image in a case, e.g. for ./dataset/CLaM/S1-A1/LCC/1.png, the ImagePath is CLaM-sample/S1-A1/LCC/1.png |
|Study_Description| Description of the exam                                                                                             |
|Views            | view of the image. Possible values: LCC, LMLO, LML, LLM, LXCCL, RCC, RMLO, RML, RLM, RXCCL.                         |
|CaseLabel        | Case-level diagnosis of malignant or benign assigned to each image in a case. Possible values: malignant or  benign.         |
|Split            | indicates whether the image is included in the train or test set. A case assigned to the train/test set has all its images assigned to the same set. Possible values: train or test.|

## How to access CLaM on the platform
The CLaM dataset on the [platform](https://fe.zgt.nl) is located under ```/mnt/dataset```. Each case can be accessed using the path ```/mnt/dataset``` + column name ```CasePath``` in ```/mnt/dataset/clam-details-case-extrainfo.csv``` and each image can be accessed using the path ```/mnt/dataset``` + column name ```ImagePath``` in ```/mnt/dataset/clam-details-image.csv```.
