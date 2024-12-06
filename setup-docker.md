# Install Docker

## Setup docker on windows
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

## Setup docker on ubuntu
We installed docker using the [apt repository](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository).
