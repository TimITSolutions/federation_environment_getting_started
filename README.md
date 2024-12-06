# The model-to-data platform

🆒 Bring the model to the (privacy-sensitive) data, instead of the data to the model.


The [model-to-data platform](https://fe.zgt.nl) provides worldwide access to our mammography dataset, [CLaM](link-to-dataset-description), in a privacy-preserving manner. The dataset was collected by the Ziekenhuis Groep Twente (ZGT) in the Netherlands. The platform allows users to submit their code to train a machine learning model on CLaM. Users upload their code for model training, the model is trained on the dataset, and users receive evaluation results and, upon request, the trained model.

<img src="data-access-platform.png" alt="platform-overview" style="height: 220px; width:800px;"/>

What can you find in this repository <br/>
:star: Instructions to write your code, upload it to the platform and view the results. <br/>
:star: Sample code on toy dataset (iris) to quickly test the upload pipeline. <br/>
:star: Sample code for breast cancer prediction on CLaM dataset to kickstart your breast cancer code development. <br/>
:star: Subset of the CLaM dataset to give you an idea about CLaM. <br/>
:star: Local docker environment similar to our platform to locally debug your code before submitting. <br/>

## Getting started
1. [Sign-up on the platform](#sign-up-on-the-platform).
2. [Test the upload pipeline with our sample code](#test-the-upload-pipeline-with-our-sample-code)
3. [Develop and test your code locally](#develop-and-test-your-code-locally)
4. [Submit your code to our platform](#submit-your-code-to-our-platform)

### Sign-up on the platform
Sign-up on the platform. An admin will manually verify and approve your account, which can take a bit of time. If you don't get a response within 3 working days, reach out to j.geerdink@zgt.nl. Upon approval, you'll be find your MLFLOW username and password that you need to add to code to track your results.

### Test the upload pipeline with our sample code
You can test the upload pipeline with the sample code on toy dataset.
1. Login to your account on the platform. Copy your MLflow credentials (username and password) and add it in line 26 and 27 of the [main.py](./sample_code/toy-dataset/main.py)
2. zip main.py and requirements.txt, e.g. as submission.zip. Make sure that main.py and requirements.txt are in the root directory of the zip.
3. Upload submission.zip on the platform.
4. You will receive email notification with status of your submission.
5. Go to [mlflow.zgt.nl](mlflow.zgt.nl) to track the progress of your experiment (after you received the email notification that execution has started).
6. Send an email to [s.pathak@utwente.nl](s.pathak@utwente.nl) for receiving your trained model. 

### Develop and test your code locally
Develop your code for CLaM. Here's our sample code for breast cancer prediction on CLaM to guide you.
1. Use subset of the CLaM dataset, [CLaM-sample](./datasets) in this repository to develop your code.
2. How to access the dataset within the code can be found here. 
3. The entrypoint of the code needs to be called ```main.py```.
4. Provide all packages needed to run your code in ```requirements.txt```.
5. Log your performance metrics to mlflow using ```mlflow.log_metrics()```.
6. Set the mlflow tracking url to [http://localhost:3001/](http://localhost:3001/).

Test your code locally on CLaM-sample using a similar docker environment as the one used in the platform.
1. Install [docker]().
2. Update ```docker-compose.yaml```: replace ```/home/dataset``` in line 31 with your local path of [datasets folder](./dataset).
3. Place your ```submission.zip``` in ```docker_scripts/```.
4. Execute the docker compose environement: ```docker compose up```
5. Track the results in mlflow through [http://localhost:3001/](http://localhost:3001/)

**Test on a machine with a NVIDIA GPU**: Use ```docker-compose.yaml``` and ```docker_scripts/execute_code```. This setup reflects the exact setup on our model-to-data platform. Set up the Nvidia container toolkit on [Ubuntu](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation) or [Windows](https://developer.nvidia.com/cuda/wsl) in order to run containers with GPU acceleration <br/>

**Test on a machine without a NVIDIA GPU**: Use ```docker-compose-without-gpu.yaml``` and ```docker_scripts/execute_code-without-gpu``` and rename them to ```docker-compose.yaml``` and ```docker_scripts/execute_code``` before running step (4) below. <br/>

### Submit your code to our platform
1. Login to your account on our [platform](https://fe.zgt.nl). 
2. Copy your auto-generated **username** and **password** for MLFlow into the ```main.py``` (e.g. line 26 and 27 of the [sample code](./sample_code/main.py)).
3. The **experiment name** of your MLFLOW experiment must be named **like your MLFLOW username**
4. Change the names of the csv files: path to the dataset. 
6. Upload your code, i.e. ```submission.zip``` to our [platform](https://fe.zgt.nl). Make sure that main.py and requirements.txt are in the root directory of the zip.
7. Track your results through [mlflow.zgt.nl](https://mlflow.zgt.nl).
8. Send an email to [s.pathak@utwente.nl](s.pathak@utwente.nl) for receiving your trained model or other log files.

## Additional Information

### MLflow
Log your performance metrics (accuracy, F1, AUC etc.) on the train and test set and also track the progress of model training at each epoch with MLflow, ```mlflow.log_metrics()```. We have disabled saving artifacts on our MLflow server to protect the privacy of our dataset. Thus, you will not be able to save your trained model to MLflow. However, you can write logs, other data and save trained models to ```/mnt/export/```. An admin can later access this volume and share the data with you on your request.

### Email notification
the code received (the docker image is bulit after this, so it will take some time), queued for execution, execution started, and execution finished.

### Run SOTA case-level breast cancer model

Case-level breast cancer model, ES-Att-Side, that works on CLaM dataset. 
- Clone the [repository](https://github.com/ShreyasiPathak/case-level-breast-cancer-data-access).
- Add the correct config file in lines 61, 94 and 100. runs/run1/config_8.ini is for testing on the platform and runs/run2/config_8.ini is for testing locally.
- Comment lines 295, 297 and 298 [here](https://github.com/ShreyasiPathak/case-level-breast-cancer-data-access/blob/main/setup/read_input_file.py) to train the model on the complete CLaM dataset.
