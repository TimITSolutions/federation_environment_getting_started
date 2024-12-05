# The model-to-data platform

:star2: Bring the model to the (privacy-sensitive) data, instead of the data to the model.


The [model-to-data platform](https://fe.zgt.nl) provides worldwide access to our mammography dataset, [CLaM](link-to-dataset-description), in a privacy-preserving manner. The dataset was collected by the Ziekenhuis Groep Twente (ZGT) in the Netherlands. The platform allows users to submit their code to train a machine learning model on CLaM. Users upload their code for model training, the model is trained on the dataset, and users receive evaluation results and, upon request, the trained model.

:star: What can you find in this repository
1. Instructions to write your code, upload it to the platform and view the results.
2. Sample code on toy dataset (iris).
3. Sample code for breast cancer prediction on CLaM dataset.
4. Subset of the CLaM dataset.
5. Local docker environment similar to our platform to locally debug your code before submitting. 

# Getting started
1. [Sign-up on the platform](#sign-up-on-the-platform).
2. [Test the upload pipeline with our sample code](#test-the-upload-pipeline-with-our-sample-code)
3. [Develop and test your code locally](#develop-and-test-your-code-locally)
4. [Submit your code to our platform](#submit-your-code-to-our-platform)

## Sign-up on the platform
Sign-up on the platform. An admin will manually verify and approve your account, which can take a bit of time. If you don't get a response within 3 working days, reach out to j.geerdink@zgt.nl. Upon approval, you'll be find your MLFLOW username and password that you need to add to code to track your results.

## Test the upload pipeline with our sample code
You can test the upload pipeline with the sample code on toy dataset.
1. Login to your account on the platform. Copy your MLflow credentials (username and password) and add it in line 26 and 27 of the [main.py](./sample_code/toy-dataset/main.py)
2. zip main.py and requirements.txt, e.g. as submission.zip. Make sure that main.py and requirements.txt are in the root directory of the zip.
3. Upload submission.zip on the platform.
4. You will receive email notification with status of your submission.
5. Go to [mlflow.zgt.nl](mlflow.zgt.nl) to track the progress of your experiment (after you received the email notification that execution has started).
6. Send an email to [s.pathak@utwente.nl](s.pathak@utwente.nl) for receiving your trained model. 

## Develop and test your code locally
Develop your code
1. Refer to the subset of the CLaM dataset, [CLaM-sample](./datasets) in this repository to develop your code.
2. The entrypoint of the code needs to be called ```main.py```.
3. Provide all packages needed to run your code in ```requirements.txt```.
4. We provide a sample code for CLaM dataset

Test your code locally using a similar docker environment as the one used in the platform.
1. Install [docker]().
2. Update ```docker-compose.yaml```: replace ```/home/dataset``` in line 31 with your local path of [datasets folder](./dataset).
4. Place your ```submission.zip``` in ```docker_scripts/```.
5. Set up the Nvidia container toolkit on [Ubuntu](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation) or [Windows](https://developer.nvidia.com/cuda/wsl) in order to run containers with GPU acceleration
6. Execute the docker compose environement: ```docker compose up```
7. Track the results in mlflow through [http://localhost:3001/](http://localhost:3001/)

**Test on a machine with a NVIDIA GPU**: Use ```docker-compose.yaml``` and ```docker_scripts/execute_code```. This setup reflects the exact setup on our model-to-data platform. <br/>
**Test on a machine without a NVIDIA GPU**: Use ```docker-compose-without-gpu.yaml``` and ```docker_scripts/execute_code-without-gpu``` and rename them to ```docker-compose.yaml``` and ```docker_scripts/execute_code``` before running step (6) below. You don't need to execute step (5) in this setting. <br/>

## Submit your code to our platform

## About this repository

This respository fulfills three main purposes:

1. The section [Running your own code on our model-to-data platform](#running-your-own-code-on-our-model-to-data-platform) describes, i) how to prepare your code to submit to our platform, ii) how to set up a local environment where you can test if your code would work on our platform, and iii) how to submit your code to our platform. 

2. The section [Running our sample code](#running-our-sample-code) contains 2 code - one simple code on iris dataset and one for a case-level breast cancer model that works on CLaM dataset. You can test these code on our platform and refer to these while preparing your code.

3. The section [Additional Information](#additional-information) provides information on how to setup docker on windows and ubuntu. 

## CLaM Dataset
<ins>C</ins>ase-<ins>La</ins>belled <ins>M</ins>ammography (CLaM) dataset contains mammography exams from Ziekenhuis Groep Twente (ZGT), The Netherlands, taken between 2013 to 2020. Our complete CLaM dataset is stored at ZGT and is not downloadable or directly accessible, but can be used for training AI models through our [platform](https://fe.zgt.nl). Details of the dataset can be found in our paper (in progress). <br/>   
We provide a sample of the CLaM dataset, [CLaM-sample](./datasets) in this repository for users to prepare their code that can work on the CLaM dataset. CLaM-sample contains 10 cases or mammography exams (S01-A01, S02-A02, ..) from 10 patients (P01, P02, ...). A mammography exam/case contains images of standard views from left (L) and right (R) breast - LMLO, LCC, RMLO and RCC and can also contain additional views - LXCCL, RXCCL, LLM, RLM, LML, RML. Each image folder, e.g. S01-A01/LMLO contains 2 images - 1.png (image in 8 bit) and 1.npy (image in 12 bit). The [datasets](./datasets) folder also contains csv files with the list of the [cases](./datasets/clam-small-subset-deceased-case.csv) and their corresponding case-level diagnosis of malignant or benign and list of [images in the cases](./datasets/clam-small-subset-deceased-image.csv). 

The CLaM dataset stored in our [platform](https://fe.zgt.nl) reflects a similar structure and can be accessed similarly.

## Running your own code on our model-to-data platform

### Prepare your code for the platform

Here are the conditions that need to be fulfilled for the submission to run successfully:

1. The entrypoint of the code needs to be called ```main.py```.  

2. A ```requirements.txt``` file needs to be provided.

3. **Username** and **password** of the MLFLOW user must be included. The username and password can be found after you login to your account on our [platform](https://fe.zgt.nl).

4. The **experiment name** of your MLFLOW experiment must be named **like your MLFLOW username**. 

5. The submission, i.e. your code must be in ```zip``` format. Export the master branch as a ZIP file:
```bash
git archive --format=zip --output submission.zip master
```
Exporting the code via ```git``` might seem like unnecessary work, but ensures that the format is correct (i.e. ```submission.zip``` should contain the ```main.py``` and ```requirements.txt``` directly under it) and the command will work on any platform (Windows, Linux, MacOS). 

6. We provide a sample of our CLaM dataset, [CLaM-sample](./datasets) in this repository. Refer to CLaM-sample while writing your code.

7. The dataset in the [model-to-data platform](https://fe.zgt.nl) is located under ```/mnt/dataset```. Each case can be accessed using the path ```/mnt/dataset``` + column name ```CasePath``` in ```/mnt/dataset/clam-details-case.csv``` and each image can be accessed using the path ```/mnt/dataset``` + column name ```ImagePath``` in ```/mnt/dataset/clam-details-image.csv```.

8. You can log your performance metrics (accuracy, F1, AUC etc.) on the train and test set and also track the progress of model training at each epoch with MLflow, ```mlflow.log_metrics()```. We have disabled saving artifacts with our MLflow instance to protect the privacy of our dataset. Thus, you will not be able to save your trained model to MLflow. However, you can write logs, other data and save trained models to ```/mnt/export/```. An admin can later access this volume and share the data with you on your request.

### Testing your code locally

As debugging submitted code becomes quite difficult, you can test on your local computer if your code will successfully work on our platform.  

**For testing on a machine with a NVIDIA GPU**: Use ```docker-compose.yaml``` and ```docker_scripts/execute_code```. This setup reflects the exact setup on our model-to-data platform. <br/>

**For testing on a machine without a NVIDIA GPU**: Use ```docker-compose-without-gpu.yaml``` and ```docker_scripts/execute_code-without-gpu``` and rename them to ```docker-compose.yaml``` and ```docker_scripts/execute_code``` before running step (7) below. You don't need to execute step (6) in this setting. <br/>

1. Install docker locally. Refer to [Additional Information](#additional-information) to setup docker.

2. **Username** and **password** of the MLFLOW user is not needed for testing locally.

3. MLFLOW output must be tracked through [localhost:3001](http://localhost:3001). Add ```mlflow.set_tracking_uri(uri="http://localhost:3001")``` to your ```main.py```.

4. In ```docker-compose.yaml``` under volumes (line 31), change ```/home/datasets``` to your local path of [datasets folder](./datasets).

5. Place your ```submission.zip``` in ```docker_scripts/```. 

6. Set up the Nvidia container toolkit on [Ubuntu](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation) or [Windows](https://developer.nvidia.com/cuda/wsl) in order to run containers with GPU acceleration

7. Execute the docker compose environement:
```bash
docker compose up
```

You can now observe the output of your submission in the terminal and view your results in mlflow through [http://localhost:3001/](http://localhost:3001/). 

### Submit your code to our platform

1. Login to your account on our [platform](https://fe.zgt.nl). If you don't have any account, sign up. Then, an admin will approve your sign up request after review.

2. Copy your auto-generated **username** and **password** for MLFlow into the ```main.py``` (e.g. line 26 and 27 of the [sample code](./sample_code/main.py)).

3. Upload your code, i.e. ```submission.zip``` to our [platform](https://fe.zgt.nl) and wait for results to become visible in [mlflow.zgt.nl](https://mlflow.zgt.nl).

## Running our sample code
We provide 2 sample code to test [locally](./README.md#testing-your-code-locally) or on our [platform](./README.md#submit-your-code-to-our-platform). You can also refer to this while preparing your code. </br> 

(1) Simple code on iris dataset </br>
- zip the [code](./sample_code) (e.g. as ```submission.zip```) and submit.

(2) Case-level breast cancer model, ES-Att-Side, that works on CLaM dataset. 
- Clone the [repository](https://github.com/ShreyasiPathak/case-level-breast-cancer-data-access).
- Add the correct config file in lines 61, 94 and 100. runs/run1/config_8.ini is for testing on the platform and runs/run2/config_8.ini is for testing locally.
- Comment lines 295, 297 and 298 [here](https://github.com/ShreyasiPathak/case-level-breast-cancer-data-access/blob/main/setup/read_input_file.py) to train the model on the complete CLaM dataset.

## Additional Information

### Setup docker on windows
For testing your code locally, you need to have docker setup on your machine. Below are the steps we followed to set it up on windows.</br>

Install docker for Windows 11 Enterprise: 

	1. Install docker desktop for windows at https://docs.docker.com/desktop/install/windows-install/. 
	2. Command in PowerShell: Start-Process 'Docker Desktop Installer.exe' -Wait install 
	3. When prompted, ensure the Use WSL 2 instead of Hyper-V option on the Configuration page is selected or not depending on your choice of backend. We used WSL-2. 
	4. Check if docker is installed successfully: docker --version 
	5. The system may be restarted or need to be restarted after this. 

If your account is different from admin account, give permissions to docker:

	1. Go to computer management (from search) and run as administrator. 
	2. Go to local users and groups node. 
	3. Click on groups folder. 
	4. Locate docker-users group in the list. 
	5. Click on add-> advanced -> find now. 
	6. Then click on your user account from the search result. We had to select authenticated users from the list. 
	7. Click Apply and ok.
    8. You can check users in docker group through these commands in PowerShell: net user or Get-LocalUser. 

Start docker desktop 

	1. Search docker desktop on the search bar. 
	2. If the permissions for your user account are correct, then clicking on docker desktop will open the app, otherwise it will not open. 
	3. Accept docker subscription service agreement. 
	4. Create an account if you don't have one. While creating an account, note that all letters should be small in username (otherwise it will show invalid format). 
	5. After creating, login with your username and password. If you login with your email address and password, then when running step 6, it will show "unauthorized: incorrect username or password". To resolve this, log out and login correctly again with your username and not email address. Also, do this in PowerShell: docker login --username your-username. Then, step 6 should work correctly. 
	6. Go to powershell and type: docker run hello-world. If this shows hello-world, then docker is successfully installed in your machine. 

### Setup docker on ubuntu
We installed docker using the [apt repository](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository).

### Email notification
the code received (the docker image is bulit after this, so it will take some time), queued for execution, execution started, and execution finished.
