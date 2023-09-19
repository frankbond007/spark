# Starting from the base image
FROM bitnami/spark:latest

# Update the repository sources list and install the desired packages
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    zip \
    unzip \
    tar \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN install_packages python3-pip
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
# Note: The 'clear' command is usually available by default in many Linux distributions
# but in case it's not, you can symlink it from `tput clear`
#RUN if ! command -v clear &> /dev/null; then ln -s $(which tput) /usr/local/bin/clear; fi
