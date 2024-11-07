# Reference code for submission to the federation environment.

## Requirements

There are a three simple conditions that need to be fulfilled for the submission to run successfully:

1. The entrypoint of the code needs to be called ```main.py```.  
2. A ```requirements.txt``` file needs to be provided.
3. **Username** and **password** of the MLFLOW user must be included.
5. The submission must be in ```zip``` format. 

The dataset in the federation environment is located under ```/mnt/dataset```. 

## Getting started

In order to test this code, perform the following steps:

1. Login to your account on [fe.zgt.nl](fe.zgt.nl).
2. Copy your auto-generated **username** and **password** into the ```main.py``` of this repository.
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

## Testing your own code

As debugging submitted code becomes quite difficult, you can test on your local computer if your code is able to be evaluated.

1. Download an ubuntu docker container, attach your ```submission.zip``` that you created earlier and open up a command line session:

```bash
# set up a docker container and attach your submission.zip file
# update the path to your submission.zip accordingly
docker run -ti -p 3001:3001 --mount type=bind,source=/path/to/submission.zip,target=/submission.zip --rm ubuntu /bin/bash
```

2. Now run the following commands:

```bash
# install dependencies
apt update && apt install -y python3 python3-pip unzip ffmpeg libsm6 libxext6 git

# install mlflow
pip3 install mlflow --break-system-packages

# unpack your submission
unzip submission.zip

pip3 install -r requirements.txt --break-system-packages

# this will start the mlflow tracking server in the background
mlflow server -p 3001 &

# start your main file
python3 main.py
```

**If you executed the above commands and you see output like this then it worked**:
```bash
[...]
Created version '1' of model 'tracking-quickstart'.
2024/11/07 17:41:30 INFO mlflow.tracking._tracking_service.client: 🏃 View run crawling-tern-354 at: http://localhost:3001/#/experiments/793410574682251178/runs/cd964aef0a1e4b68a32376ecbf0d9a8c.
2024/11/07 17:41:30 INFO mlflow.tracking._tracking_service.client: 🧪 View experiment at: http://localhost:3001/#/experiments/793410574682251178.
```
