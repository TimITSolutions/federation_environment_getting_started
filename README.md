# The model-to-data platform

🆒 Bring the model to the (privacy-sensitive) data, instead of the data to the model.


The [model-to-data platform](https://fe.zgt.nl) provides worldwide access to our mammography dataset, [CLaM](./dataset.md), in a privacy-preserving manner. The dataset was collected from the Ziekenhuis Groep Twente (ZGT) in the Netherlands. The platform allows users to submit their code to train a machine learning model on CLaM. Users upload their code for model training, the model is trained on the dataset, and users receive evaluation results and, upon request, the trained model.

<img src="data-access-platform.png" alt="platform-overview" style="height: 220px; width:800px;"/>

What can you find in this repository? <br/>
:star: Instructions to write your code, upload it to the platform and view the results. <br/>
:star: Sample code on toy dataset (iris) to quickly test the upload pipeline. <br/>
:star: Sample code for breast cancer prediction on CLaM dataset to kickstart your breast cancer code development. <br/>
:star: Subset of the CLaM dataset to give you an idea about CLaM. <br/>
:star: Local docker environment similar to our platform to locally debug your code before submitting. <br/>

## Getting started
Clone this repository and follow the steps below to get started.
1. [Sign-up on the platform](#sign-up-on-the-platform)
2. [Test the upload pipeline with our sample code](#test-the-upload-pipeline-with-our-sample-code)
3. [Develop and test your code locally](#develop-and-test-your-code-locally)
4. [Submit your code to our platform](#submit-your-code-to-our-platform)

### Sign-up on the platform
Sign-up on the platform. An admin will manually verify and approve your account, which can take a bit of time. If you don't get a response within 3 working days, reach out to Jeroen Geerdink ([j.geerdink@zgt.nl](j.geerdink@zgt.nl)). Upon approval, you will be assigned an MLflow username and password in your account. You need to add this to your code to track your results.

### Test the upload pipeline with our sample code
You can test the upload pipeline with our [sample code on toy dataset](./sample_code/toy-dataset).
1. Login to your account on the platform. Copy your MLflow credentials (username and password) and add it in lines 26 and 27 of [main.py](./sample_code/toy-dataset/main.py)
2. zip ```main.py``` and ```requirements.txt```, e.g. as ```submission.zip```. Make sure that ```main.py``` and ```requirements.txt``` are in the root directory of the zip.
3. Upload ```submission.zip``` on the platform.
4. You will receive an [email notification with status](#email-notification) of your submission and errors in code if encountered.
5. Go to [mlflow.zgt.nl](https://mlflow.zgt.nl) to track the progress of your experiment (after you received the email notification that execution has started).
6. On successful execution of the code, you will see accuracy = 1 and dataset-exists = 1 under metrics in MLflow.

### Develop and test your code locally
We suggest that you bootstrap development from our [sample code on CLaM dataset](./sample_code/clam-dataset). This sample code can seamlessly be tested locally and submitted to the platform. It trains a standard ResNet for breast cancer prediction at the image-level on our CLaM dataset. For local testing, we included a small subset of CLaM, [CLaM-sample](./dataset) in this repository. 

However, if you would like to start from scratch, most important things to keep in mind when developing your code for CLaM are:
1. Use [CLaM-sample](./dataset) in this repository to develop your code.
2. CLaM dataset description and how to access the dataset in your code can be found [here](./dataset.md). 
3. The entrypoint of the code needs to be called ```main.py```.
4. Provide all packages needed to run your code in ```requirements.txt``` ([generate requirements.txt](#generate-requirementstxt)). 
5. Log your performance metrics to MLflow using ```mlflow.log_metrics()```.
6. Log your output files and trained model in ```/mnt/export``` ([more explanation](#mlflow)). 
7. Set the MLflow tracking url to [http://localhost:3001/](http://localhost:3001/).


Test your code locally on CLaM-sample in a similar docker environment as the one used in the platform.
1. Install [docker](./setup-docker.md).
2. Update ```docker-compose.yaml```: replace ```/home/dataset``` in line 31 with your local path of the [dataset folder](./dataset).
3. Place your ```submission.zip``` in ```docker_scripts/```.
4. Execute the docker compose environment: ```docker compose up```
5. Track the results in MLflow through [http://localhost:3001/](http://localhost:3001/)
6. Error during local testing? [Refer to this page for some useful tips](useful-docker-commands.md).

**Test on a machine with a NVIDIA GPU**: Correct files are ```docker-compose.yaml``` and ```docker_scripts/execute_code```. This setup reflects the exact setup on our model-to-data platform. Set up the Nvidia container toolkit on [Ubuntu](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation) or [Windows](https://developer.nvidia.com/cuda/wsl) before running step (4) in order to run containers with GPU acceleration. <br/>

**Test on a machine without a NVIDIA GPU**: Correct files are ```docker-compose-without-gpu.yaml``` and ```docker_scripts/execute_code-without-gpu```. Rename them to ```docker-compose.yaml``` and ```docker_scripts/execute_code``` before running step (4) below. <br/>

### Submit your code to our platform
1. Login to your account on our [platform](https://fe.zgt.nl). 
2. Copy your auto-generated **username** and **password** for MLflow into the ```main.py``` (e.g. lines 302 and 303 of the [sample code](./sample_code/clam-dataset/main.py)).
3. The **experiment name** of your MLflow experiment must be named **like your MLflow username**  
4. Upload your zip code, i.e. ```submission.zip``` to our [platform](https://fe.zgt.nl). Make sure that ```main.py``` and ```requirements.txt``` are in the root directory of the zip.
5. Track your results through [mlflow.zgt.nl](https://mlflow.zgt.nl).
6. Send an email to [s.pathak@utwente.nl](s.pathak@utwente.nl) for receiving your trained model or other log files.

## Additional Information

### MLflow
Log your performance metrics (accuracy, F1, AUC etc.) on the train and test set and also track the progress of model training at each epoch with MLflow, ```mlflow.log_metrics()```. We have disabled saving artifacts on our MLflow server to protect the privacy of our dataset. Thus, you will not be able to save your trained model to MLflow. However, you can write logs, other data and save trained models to ```/mnt/export/```. An admin can later access this volume and share the data with you on your request.

### Email notification
You will receive email notification with the status of your code. 
- ```code received```. The docker image is bulit after this, so it will take some time after this stage
- ```queued for execution```
- ```execution started```
- ```execution finished or failed```. If your code failed to run, then you will also get the error with the email notification.

### Generate requirements.txt

- ```pip install pipreqs```
- ```pipreqs /path/to/project```

### Run SOTA case-level breast cancer model

Case-level breast cancer model, ES-Att-Side, that works on CLaM dataset. 
- Clone the [repository](https://github.com/ShreyasiPathak/case-level-breast-cancer-data-access).
- Add the correct config file in lines 61, 94 and 100. ```runs/run1/config_8.ini``` is for testing on the platform and ```runs/run2/config_8.ini``` is for testing locally.
- Comment lines 295, 297 and 298 [here](https://github.com/ShreyasiPathak/case-level-breast-cancer-data-access/blob/main/setup/read_input_file.py) to train the model on the complete CLaM dataset.
