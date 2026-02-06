#!/bin/bash -xe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PARENT_DIR/ros2_ws" || exit 1
source /opt/ros/humble/setup.bash

sudo rm -rf build/ install/ log/
colcon build --symlink-install

source install/setup.bash

echo -e "\e[92m\n\e[1m Installed ros2_ws. \n\e[0m"