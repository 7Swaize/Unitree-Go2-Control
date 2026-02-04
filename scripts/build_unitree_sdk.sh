#!/bin/bash -xe

################################################################################
# UniTree SDK2 Python Installation Script for Jetson Orin Nano
# 
# Usage: ./install_unitree_sdk [OPTIONS]
#
# Options:
#   --install-dir <path>        Installation directory (default: ~/librealsense_build)
################################################################################


INSTALL_DIR="$HOME"

patch_cyclonedds () {
    sudo apt-get install -y curl lsb-release gnupg
    sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y gz-harmonic
}


while [[ $# -gt 0 ]]; do
    case $1 in
        --install-dir) 
            [[ -z "$2" ]] && { echo "Error: --install-dir requires a value"; exit 1; }
            INSTALL_DIR=$2
            shift 2 ;;
        *) echo "Error: Unknown option: $1"; exit 1 ;;
    esac
done


sudo apt-get update
sudo apt-get install -y python3-pip 

cd "$INSTALL_DIR"
if [ ! -d "unitree_sdk2_python" ]; then
    git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
fi
cd unitree_sdk2_python

set +e
pip3 install -e . 2>&1 | tee tmp_install.log
INSTALL_STATUS=${PIPESTATUS[0]}
set -e

if grep -i "cyclonedds" tmp_install.log; then
    echo -e "\e[31m Detected CycloneDDS error, applying patch... \033[0m"
    patch_cyclonedds
    sudo rm tmp_install.log
    pip3 install -e .
fi


echo -e "\e[92m\n\e[1m  UniTree SDK2 Python installation completed. \n\e[0m"
