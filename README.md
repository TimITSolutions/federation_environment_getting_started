# Our model-to-data platform and instructions on how to use it
We developed a [model-to-data platform](fe.zgt.nl) to provide users worldwide access to our mammography dataset, CLaM, in a privacy preserving manner. The dataset is collected from Ziekenhuis Groep Twente (ZGT), The Netherlands. Through our model-to-data platform, users can send their artificially intelligent (AI) model script for training on our mammography dataset. The mammography dataset stays within the hospital and users cannot download or see the dataset. The users only get back the model’s performance on our dataset when the training is complete and can also get the trained model on request. Thus, instead of bringing the data to the model, we bring the model to the data.


## About this repository

This respository fulfills three main purposes:

1. The section [Running your own code on our model-to-data platform](#running-your-own-code-on-our-model-to-data-platform) describes, i) how to prepare your code to submit to our platform, ii) how to set up a local environment, in which you can test if your code would work on our platform, and iii) how to submit your code to our platform. 

2. The section [Running our sample code](#running-our-sample-code) contains 2 code - one simple code on iris dataset and one for a case-level breast cancer model that works on CLaM dataset. You can test these code on our platform and refer to these while preparing your code.

3. The section [Additional Information](#additional-information) provides information on how to setup docker on windows and ubuntu. 

## CLaM Dataset
<ins>C</ins>ase-<ins>La</ins>belled <ins>M</ins>ammography (CLaM) dataset contains mammography exams from Ziekenhuis Groep Twente (ZGT), The Netherlands, taken between 2013 to 2020. Our complete CLaM dataset is stored at ZGT and is not downloadable or directly accessible, but can be used for training AI models through our [platform](fe.zgt.nl). Details of the dataset can be found in our paper (in progress). <br/>   
We provide a sample of the CLaM dataset, [CLaM-sample](./datasets) in this repository for users to prepare their code that can work on the dataset. CLaM-sample contains 10 cases or mammography exams (S01-A01, S02-A02, ..) from 10 patients (P01, P02, ...). A mammography case contains images from standard views - LMLO, LCC, RMLO and RCC and can also contain additional views - LXCCL, RXCCL, LLM, RLM, LML, RML. Each image folder, e.g. S01-A01/LMLO contains 2 images - 8.png (image in 8 bit) and 16.png (image in 16 bit). The [datasets](./datasets) folder also contains csv files with the list of the cases and their corresponding diagnosis of malignant or benign. 

## Running your own code on our model-to-data platform

### Prepare your code for the platform

There are a four simple conditions that need to be fulfilled for the submission to run successfully:

1. The entrypoint of the code needs to be called ```main.py```.  
2. A ```requirements.txt``` file needs to be provided.
3. **Username** and **password** of the MLFLOW user must be included.
4. The **experiment name** of your MLFLOW experiment must be named **like your MLFLOW username**. 
5. The submission must be in ```zip``` format.
6. We provide a sample of our CLaM dataset, [CLaM-sample](./datasets) in this repository. Refer to CLaM-sample while writing your code. 

The dataset in the model-to-data platform is located under ```/mnt/dataset```. 

You can write logs, other data and save trained models to ```/mnt/export/```, a staff member can later access this volume and share the data with you.

### Testing your code locally

As debugging submitted code becomes quite difficult, you can test on your local computer if your code will successfully work on our platform. Note: This step requires you to have a NVIDIA GPU in your system

Before you can follow the steps below, you need to have docker setup locally. Refer to [Additional Information](#additional-information) to setup docker. 

1. **Username** and **password** of the MLFLOW user is not needed for testing locally.

2. Place your ```submission.zip``` in ```docker_scripts/```. 

3. Set up the [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation) in order to run containers with GPU acceleration

4. Execute the docker compose environement:
```bash
docker compose up
```

You can now observe the output of your submission in the terminal. 

You can also check the MLFLOW output in [localhost:3001](localhost:3001).

### Submit your code to our platform

In order to test this code, perform the following steps:

1. Login to your account on [fe.zgt.nl](fe.zgt.nl).
2. Copy your auto-generated **username** and **password** into the ```main.py``` of this repository (line 26 and 27 in respectively).
3. Commit your changes:
```bash
git commit -am "added my own credentials"
```
4. Export the master branch as a ZIP file:
```bash
git archive --format=zip --output submission.zip master
```
5. Upload the code to [fe.zgt.nl](fe.zgt.nl) and wait for results to become visible in [mlflow.zgt.nl](mlflow.zgt.nl).

Exporting the code via ```git``` might seem like unnecessary work, but ensures that the format is correct and the command will work on any platform (Windows, Linux, MacOS).

## Running our sample code
We provide 2 sample code to test [locally](./README.md#testing-your-code-locally) or [on our platform](./README.md#submit-your-code-to-our-platform). You can also refer to this while preparing your code. </br> 

(1) [Simple code on iris dataset](./sample_code) </br>

(2) Case-level breast cancer model, [ES-Att-Side](https://github.com/ShreyasiPathak/case-level-breast-cancer-data-access), that works on CLaM dataset. Clone the repository and get started.

## Additional Information

### Setup docker on windows
For testing your code locally, you need to have docker setup on your machine. Below are the steps we followed to set it up on windows.</br>

Install docker for Windows 11 Enterprise: 

	1. Install [docker desktop for windows]( https://docs.docker.com/desktop/install/windows-install/). 
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

