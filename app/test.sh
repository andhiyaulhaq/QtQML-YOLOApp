#!/bin/bash
echo "Building tests..."
cmake --build build --config Release -j 4

if [ $? -ne 0 ]; then
    echo "Build failed. Tests will not be executed."
    exit 1
fi

echo "Running tests..."
cd build && ctest -C Release --output-on-failure
