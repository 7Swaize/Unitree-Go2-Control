#!/bin/bash -xe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# bash "$SCRIPT_DIR/build_realsense.sh"
bash "$SCRIPT_DIR/build_unitree_sdk.sh"
bash "$SCRIPT_DIR/build_c_extensions.sh"
bash "$SCRIPT_DIR/build_ros_ws.sh"


echo -e "\e[92m\n\e[1m Installed all native dependencies. \n\e[0m"
echo "Next steps:"
echo "  1. Reboot system"
echo "  2. Plug in RealSense camera"
echo "  3. Test with: realsense-viewer"
