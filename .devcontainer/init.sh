#!/bin/bash
IPYTHON_PROFILE_DIR=~/.ipython/profile_default
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p ${IPYTHON_PROFILE_DIR}
cp ${SCRIPT_DIR}/ipython_config.py ${IPYTHON_PROFILE_DIR}/

