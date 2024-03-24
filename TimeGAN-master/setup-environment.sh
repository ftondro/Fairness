#!/bin/bash

# NOTE TO ADMIN: dependencies on lines 25-29 require sudo access.

# Install pyenv if it's not installed
if [ ! -d "$HOME/.pyenv" ]; then
    echo "Installing pyenv"
    wget -q https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer -O- | bash
fi
else
    echo "pyenv is already installed"
fi

# Add pyenv to bash so it can be run
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Set pyenv to PATH so it loads every time you open a terminal
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc

# Install Python 3.7 if it's not installed
if [ ! -d "$HOME/.pyenv/versions/3.7.17" ]; then
    # Install necessary libraries for building Python 3.7.17 (Only necessary if you don't have Python 3.7.17 installed)
    echo "Installing necessary libraries for building Python"
    sudo apt-get update
    sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

    echo "Installing Python 3.7.17"
    pyenv install 3.7.17
fi
else
    echo "Python 3.7.17 is already installed"
fi

# Create a new virtualenv with Python 3.7 if it's not created
if [ ! -d "$HOME/.pyenv/versions/3.7.17/envs/my_project_env" ]; then
    echo "Creating a new virtualenv with Python 3.7.17"
    pyenv virtualenv 3.7.17 my_project_env
fi
else
    echo "Virtualenv my_project_env is already created"
fi

# Activate the virtualenv
pyenv activate my_project_env
echo "Activated Virtualenv my_project_env"

# Set the local version to be used in your project
pyenv local my_project_env

# Install the requirements
echo "Installing the requirements"
pip install -r requirements.txt