# Librealsense SDK - RSUSB Build
 
The Jetson Orin Nano is unique, so the source files must be manually built. This process takes approximately 30 minutes. CUDA GPU acceleration is optional — the RSUSB build is simpler and safer.

Installation directions adapted from: [Librealsense Documentation](https://github.com/realsenseai/librealsense/blob/master/doc/installation_jetson.md)

## Setup

Install python dependencies.

```bash
conda activate <env_name>
conda install conda-forge::python-devtools
```

Note the path to your Python interpreter — you will need it during the build:

```bash
which python
```
 
## CUDA Configuration (Optional)
 
Check if the CUDA compiler is available:
 
```bash
nvcc --version
```
 
If the "Cuda compiler driver" is not shown, you can either build without CUDA support or install the tools manually. 

### Installing the CUDA Toolkit

Install the NVIDIA JetPack toolkit:

```bash
sudo apt install nvidia-jetpack
```

Verify the CUDA installation directory:

```bash
ls /usr/local | grep cuda
```
 
The expected output should be something like `cuda-12.6`.
 
Add the CUDA compiler and runtime library paths to `~/.bashrc`:

```bash
# Find CUDA compiler and link CUDA libs at runtime
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```
 
Test that `nvcc` is recognized and outputs the correct version information:
```bash
nvcc --version
```


## Build
 
Download the build script:

```bash
cd ~ && wget https://raw.githubusercontent.com/realsenseai/librealsense/refs/heads/master/scripts/libuvc_installation.sh -O libuvc_installation.sh
```
 
Edit the build script to configure the CMake invocation. Replace `<python_interpreter>` with the path from the `which python` output above.

**With CUDA** — also replace `<cuda_compiler>` with the path to your CUDA compiler (e.g. `/usr/local/cuda-12.6/bin/nvcc`):

```bash
cmake ../ -DFORCE_LIBUVC=true -DBUILD_PYTHON_BINDINGS:bool=true \
        -DPYTHON_EXECUTABLE=<python_interpreter> \
        -DCMAKE_BUILD_TYPE=release \
        -DBUILD_WITH_CUDA=true \
        -DCMAKE_CUDA_COMPILER=<cuda_compiler>
```

**Without CUDA**:

```bash
cmake ../ -DFORCE_LIBUVC=true -DBUILD_PYTHON_BINDINGS:bool=true \
        -DPYTHON_EXECUTABLE=<python_interpreter> \
        -DCMAKE_BUILD_TYPE=release \
        -DBUILD_WITH_CUDA=false
```
 
Unplug the camera, then run the build script:
```bash
bash libuvc_installation.sh
```
 
Once the build completes, reboot, plug in the camera, and run the folloing command to test the installation. The realsense viewer window should open.

```bash
realsense-viewer
```

To enable Python support, install the Python bindings in your conda environment:
```bash
pip install pyrealsense2
```