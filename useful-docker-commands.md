# Debugging in your local docker environment

After running ```docker compose up```, if you change your code to fix errors, the changes will not get automatically reflected the next time you run ```docker compose up``` <br/>

You can do any of the following two solutions:

1. Run from scratch: ```docker-compose up --build submission_run``` or
2. Copy your new files to your container.
    1. Look up your stopped container id: ```docker ps -a```
    2. For copying the main.py: docker cp your-local-path-to-your-main.py container-id:/app/main.py
    3. docker compose up

Your code has run successfully when you get ```code_submission exited with code 0``` on your terminal. ```code_submission exited with code 1``` indicates an error.


