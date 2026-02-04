#!/bin/bash -xe

################################################################################
# Librealsense SDK Installation Script for Jetson Orin Nano
# 
# Usage: ./install_librealsense_jetson.sh [OPTIONS]
#
# Options:
#   --install-dir <path>        Installation directory (default: ~/librealsense_build)
#   --python-path <path>        Path to Python interpreter (default: auto-detect)
#   --cuda-compiler <path>      Path to CUDA compiler nvcc (default: auto-detect)
#   --enable-cuda               Build with CUDA support (default: auto-detect)
#   --disable-cuda              Build without CUDA support
#   --help                      Display this help message
################################################################################


INSTALL_DIR="~/librealsense_build"
PYTHON_PATH=""
CUDA_COMPILER=""
ENABLE_CUDA=-1  # -1: auto, 0: disable, 1: enable


while [[ "$# -gt -0"]]; do
    case $1 in 
        --install-dir) 
            [[ -z "$2" ]] && { echo "Error: --install-dir requires a value"; exit 1; }
            INSTALL_DIR=$2
            shift 2 ;;
        --python-path)
            [[ -z "$2" ]] && { echo "Error: --python-path requires a value"; exit 1; }
            PYTHON_PATH=$2
            shift 2 ;;
        --cuda-compiler) 
            [[ -z "$2" ]] && { echo "Error: --cuda-compiler requires a value"; exit 1; }
            CUDA_COMPILER=$2
            shift 2 ;;
        --enable-cuda) ENABLE_CUDA=1; shift ;;
        --disable-cuda) ENABLE_CUDA=0; shift ;;
        *) echo "Error: Unknown option: $1"; exit 1 ;;
    esac
done


# Locally suppress stderr to avoid raising not relevant messages
exec 3>&2
exec 2> /dev/null
con_dev=$(ls /dev/video* | wc -l)
exec 2>&3

if [ $con_dev -ne 0 ];
then
	echo -e "\e[32m"
	read -p "Remove all RealSense cameras attached. Hit any key when ready"
	echo -e "\e[0m"
fi

lsb_release -a
echo "Kernel version $(uname -r)"


if [[ -z "$PYTHON_PATH" ]]; then
    PYTHON_PATH=$(which python 2>/dev/null || which python3 2>/dev/null || { echo "Error: Could not auto-detect Python. Use --python-path"; exit 1; })
fi

echo "Python: $PYTHON_PATH ($($PYTHON_PATH --version 2>&1))"


if [[ $ENABLE_CUDA -eq -1 ]]; then
    if command -v nvcc &> /dev/null; then
        ENABLE_CUDA=1
    else
        ENABLE_CUDA=0
    fi
fi

if [[ $ENABLE_CUDA -eq 1 ]]; then
    if [[ -z "$CUDA_COMPILER" ]]; then 
        CUDA_COMPILER=$(which nvcc 2>/dev/null || echo "")
        if [ -z "$CUDA_COMPILER" ]; then
            for cuda_path in /usr/local/cuda-*/bin/nvcc; do
                if [ -f "$cuda_path" ]; then
                    CUDA_COMPILER="$cuda_path"
                    break
                fi
            done
        fi
        if [[ -z "$CUDA_COMPILER" ]]; then
            echo "Error: CUDA compiler not found. Use --cuda-compiler or --disable-cuda"
            exit 1
        fi

    if [ ! -f "$CUDA_COMPILER" ]; then
        echo "Error: CUDA compiler not found at $CUDA_COMPILER"
        exit 1
    fi

    echo "CUDA: $CUDA_COMPILER ($($CUDA_COMPILER --version | grep release | awk '{print $5}' | sed 's/,//'))"
else
    echo "Building without CUDA support"
fi


sudo apt-get update
sudo rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

if [ $(sudo swapon --show | wc -l) -eq 0 ];
then
	echo "No swapon - setting up 1Gb swap file"
	sudo fallocate -l 2G /swapfile
	sudo chmod 600 /swapfile
	sudo mkswap /swapfile
	sudo swapon /swapfile
	sudo swapon --show
fi

echo Installing Librealsense-required dev packages
sudo apt-get install git cmake libssl-dev freeglut3-dev libusb-1.0-0-dev pkg-config libgtk-3-dev unzip -y

rm -f ./master.zip
wget https://github.com/realsenseai/librealsense/archive/master.zip
unzip ./master.zip -d .
cd ./librealsense-master

echo Install udev-rules
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/ 
sudo cp config/99-realsense-d4xx-mipi-dfu.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger 

mkdir build && cd build
CMAKE_CMD="cmake ../ -DFORCE_LIBUVC=true -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=$PYTHON_PATH -DCMAKE_BUILD_TYPE=release"

if [[ $ENABLE_CUDA -eq 1]]; then
    CMAKE_CMD="$CMAKE_CMD -DBUILD_WITH_CUDA=true -DCMAKE_CUDA_COMPILER=$CUDA_COMPILER"
else
    CMAKE_CMD="$CMAKE_CMD -DBUILD_WITH_CUDA=false"
fi

echo "Configuring build..."
eval $CMAKE_CMD

echo "Building (this takes time)..."
make -j2
sudo make install
sudo ldconfig


echo -e "\e[92m\n\e[1mLibrealsense installation completed.\n\e[0m"
echo "Next steps:"
echo "  1. Reboot system"
echo "  2. Plug in RealSense camera"
echo "  3. Test with: realsense-viewer"