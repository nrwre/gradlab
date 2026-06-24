#!/usr/bin/env bash
# Vendors the engine into the backend folder so SAM packages it with the function.
set -e
cd "$(dirname "$0")"
rm -rf nanograd
cp -r ../engine/nanograd ./nanograd
echo "Vendored engine into backend/nanograd"
