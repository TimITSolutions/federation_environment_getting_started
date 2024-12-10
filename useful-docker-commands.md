# Debugging in your local docker environment

On running ```docker compose up```, ```code_submission exited with code 0``` on your terminal shows your code has run successfully, and ```code_submission exited with code 1``` indicates an error.

You can inspect your log files and trained model with the following command:
1. ```docker volume ls``` will show your local volume with the name ```federation_environment_getting_started_export```.
2. ```docker volume inspect federation_environment_getting_started_export``` will show the path (Mountpoint).
3. In linux system, access this path to get your log files. In windows system, access your volume through docker desktop.
4. Our sample code on CLaM dataset will generate error.txt, which logs the errors and out.txt (logs the print statements in your code).

After fixing your error, running ```docker compose up``` again will not automatically use your changed code.

You can do one of the following things to run your new code in the local docker setup:

1. Run the docker compose from scratch: ```docker-compose up --build submission_run``` or
2. Copy your new files to your container.
    1. Look up your stopped container id: ```docker ps -a```
    2. For copying the main.py: docker cp your-local-path-to-your-main.py container-id:/app/main.py
    3. docker compose up

Happy debugging!
