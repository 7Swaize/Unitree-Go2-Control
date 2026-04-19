# Unitree SDK2 Python - Installation

Installation directions adapted from: [Unitree SDK2 Python README](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/README.md)

## Clone the repository

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```

## CycloneDDS path error

If your installation succeeded, continue to the C++ installation.

You may encounter the following error during `pip install`:

> Could not locate cyclonedds. Try to set `CYCLONEDDS_HOME` or `CMAKE_PREFIX_PATH`.

As the name suggests, this means the compiler cannot find the core CycloneDDS library.

The steps below show how to install CycloneDDS from source. This is also required for shared memory support via Iceoryx.

If you have already installed CycloneDDS from source:
- Set `CYCLONEDDS_HOME` to the install directory
- Set `CMAKE_PREFIX_PATH` to the same directory, so the compiler knows where to find the installation

These instructions install CycloneDDS in the home directory. However, you can clone and install the repository anywhere you like, granted you update any paths accordingly. 


Clone the CycloneDDS repository. The Unitree SDK requires **CycloneDDS version 0.10.2**.

```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x 
mkdir -p ~/cyclonedds/build ~/cyclonedds/install
cd ~/cyclonedds/build
```

Install.

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
```

Export paths.

```bash
export CYCLONEDDS_HOME=$HOME/cyclonedds/install
export CMAKE_PREFIX_PATH=$CYCLONEDDS_HOME
```

Reinstall the Python Unitree SDK2 via pip install as outlined before. Your installation should succeed.

---

# Unitree SDK2 C++ - Installation

Installation directions adapted from: [Unitree SDK2 README](https://github.com/unitreerobotics/unitree_sdk2/blob/main/README.md)

## Environment Setup

Before building or running the SDK, ensure the following dependencies are installed:

- CMake (version 3.10 or higher)
- GCC (version 9.4.0)
- Make

You can install the required packages on Ubuntu 20.04 with:

```bash
apt-get update
apt-get install -y cmake g++ build-essential libyaml-cpp-dev libeigen3-dev libboost-all-dev libspdlog-dev libfmt-dev
```

## Installation

Clone the repository.

```bash
https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
```

Build.

```bash
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics
sudo make install
```

If you install the library in a location other than `/opt/unitree_robotics`, you will need to update the CMakeLists.txt when installing the Mujoco Simulator. 
Specifically, in [this](https://github.com/7Swaize/unitree_mujoco/blob/main/simulate/CMakeLists.txt) file, edit the following line to point to the correct install location:

```bash
list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")
```


