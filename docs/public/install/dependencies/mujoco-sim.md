# Mujoco Simulator Installation

This installation consists of two steps: first, installing the shared Iceoryx2 configuration files, and second, installing the actual simulator.


## Clone the Repository

```bash
git clone https://github.com/7Swaize/unitree_mujoco.git
cd unitree_mujoco
```

## Shared Iceoryx2 Configuration Installation

This installation must be done before installing the actual simulator.

Set `CMAKE_PREFIX_PATH` to the installed **iceoryx2-cxx** location.

If you followed the default installation layout, the path will typically be: `<containing_folder>/iceoryx2/target/ff/cc/install`

To set the environment variable, run the following command in your terminal:

```bash
export CMAKE_PREFIX_PATH=<containing_folder>/iceoryx2/target/ff/cc/install
```

Install via CMake.

```bash
cd iceoryx_interfaces
cmake -S . -B build
cmake --build build
sudo cmake --install build
```

Install via pip.

```bash
pip install .
```


## Mujoco Simulator Installation

Install.

```bash
cd simulate
pip install .
```

(Optional) Install with verbose logging.

```bash
cd simulate
pip install . -v --config-settings=logging.level=INFO
```

Run the following command in the terminal to test the installation. It should launch the simulator.
```bash
go2sim
```