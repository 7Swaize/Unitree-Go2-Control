#!/bin/bash -xe

DIR=$(dirname "${BASH_SOURCE[0]}")

bash "$DIR/build_realsense.sh"
bash "$DIR/build_unitree_sdk.sh"

echo -e "\e[92m\n\e[1m Installed all native dependencies. \n\e[0m"
echo "Next steps:"
echo "  1. Reboot system"
echo "  2. Plug in RealSense camera"
echo "  3. Test with: realsense-viewer"
