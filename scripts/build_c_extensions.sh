#!/bin/bash -xe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PARENT_DIR/c_extensions"
DEBUG=0 pip install -e .


echo -e "\e[92m\n\e[1m Installed c_extensions. \n\e[0m"