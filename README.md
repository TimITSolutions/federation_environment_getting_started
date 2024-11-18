# Reference code for submission to our model-to-data platform.

## This repository

This respository fulfills three main purposes:

1. The section [Requirements](#requirements) explains the basic requirements for the submitted sourcecode to the federation environment.

2. The section [Getting Started](#getting-started) shows you how to export your code in the correct format.

3. The section [Testing your own code](#testing-your-own-code) shows you how to set up a local environment, in which you can test if your code would work. Note: This step requires you to have a NVIDIA GPU in your system.

## Running your own code on our platform

### Requirements

There are a four simple conditions that need to be fulfilled for the submission to run successfully:

1. The entrypoint of the code needs to be called ```main.py```.  
2. A ```requirements.txt``` file needs to be provided.
3. **Username** and **password** of the MLFLOW user must be included.
4. The **experiment name** of your MLFLOW experiment must be named **like your MLFLOW username**. 
5. The submission must be in ```zip``` format. 

The dataset in the federation environment is located under ```/mnt/dataset```. 

You can write logs or other data to ```/mnt/export/```, a staff member can later access this volume and share the data with you.

### Testing your own code locally

As debugging submitted code becomes quite difficult, you can test on your local computer if your code will successfully work on fe.zgt.nl.

1. We provide a sample of our CLaM dataset, [CLaM-sample](./datasets) in this repository. Refer to CLaM-sample while writing your code. 

2. For testing locally, the dataset path in your code should be your local path to CLaM-sample.

3. Place your ```submission.zip``` in ```docker_scripts/```. 

4. Set up the [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation) in order to run containers with GPU acceleration

5. Execute the docker compose environement:
```bash
docker compose up
```

You can now observe the output of your submission in the terminal. 

### Submit your own to our model-to-data platform

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



You can also check the MLFLOW output in [localhost:3001](localhost:3001).

## Running our sample code

1. Clone the repository: https://github.com/ShreyasiPathak/case-level-breast-cancer-data-access
2. 
