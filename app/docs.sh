#!/bin/bash
# Clean previous output to prevent Graphviz caching errors
rm -rf build/docs_output

# Generate the documentation using the CMake doc target
cmake --build build --target doc

# Open the documentation
start build/docs_output/html/index.html
