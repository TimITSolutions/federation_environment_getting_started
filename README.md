# Reference code for submission to our model-to-data platform.

## This repository

This respository fulfills three main purposes:

1. The section [Requirements](#requirements) explains the basic requirements for the submitted sourcecode to the federation environment.

2. The section [Getting Started](#getting-started) shows you how to export your code in the correct format.

3. The section [Testing your own code](#testing-your-own-code) shows you how to set up a local environment, in which you can test if your code would work. Note: This step requires you to have a NVIDIA GPU in your system.

## Running your own code on our model-to-data platform

### Prepare your code for the platform

There are a four simple conditions that need to be fulfilled for the submission to run successfully:

1. The entrypoint of the code needs to be called ```main.py```.  
2. A ```requirements.txt``` file needs to be provided.
3. **Username** and **password** of the MLFLOW user must be included.
4. The **experiment name** of your MLFLOW experiment must be named **like your MLFLOW username**. 
5. The submission must be in ```zip``` format.
6. We provide a sample of our CLaM dataset, [CLaM-sample](./datasets) in this repository. Refer to CLaM-sample while writing your code. 

The dataset in the federation environment is located under ```/mnt/dataset```. 

You can write logs or other data to ```/mnt/export/```, a staff member can later access this volume and share the data with you.

### Testing your code locally

As debugging submitted code becomes quite difficult, you can test on your local computer if your code will successfully work on our platform.

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

	1. Install [docker desktop for windows]( https://docs.docker.com/desktop/install/windows-install/). <br/>
	2. Command in PowerShell: Start-Process 'Docker Desktop Installer.exe' -Wait install <br/>
	3. When prompted, ensure the Use WSL 2 instead of Hyper-V option on the Configuration page is selected or not depending on your choice of backend. We used WSL-2. <br/>
	4. Check if docker is installed successfully: docker --version <br/>
	5. The system may be restarted or need to be restarted after this. <br/>

If your account is different from admin account, give permissions to docker:

	1. Go to computer management (from search) and run as administrator. <br/>
	2. Go to local users and groups node. <br/>
	3. Click on groups folder. <br/>
	4. Locate docker-users group in the list. <br/>
	5. Click on add-> advanced -> find now. <br/>
	6. Then click on your user account from the search result. We had to select authenticated users from the list. <br/>
	7. Click Apply and ok. <br/>
        8. You can check users in docker group through: ```net user``` or ```Get-LocalUser```. 

Start docker desktop 

	1. Search docker desktop on the search bar. 
	2. If the permissions for your user account are correct, then clicking on docker desktop will open the app, otherwise it will not open. 
	3. Accept docker subscription service agreement. 
	4. Create an account if you don't have one. While creating an account, note that all letters should be small in username (otherwise it will show invalid format). 
	5. After creating, login with your username and password. If you login with your email address and password, then when running step 6, it will show "unauthorized: incorrect username or password". To resolve this, log out and login correctly again with your username and not email address. Also, do this in PowerShell: docker login --username your-username. Then, step 6 should work correctly. 
	6. Go to powershell and type: ```docker run hello-world```. If this shows hello-world, then docker is successfully installed in your machine. 

