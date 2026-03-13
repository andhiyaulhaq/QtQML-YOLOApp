#!/bin/bash
C:/Qt/6.8.3/msvc2022_64/bin/windeployqt.exe --release --qmldir content --qmldir . --dir build/Release/libs build/Release/appCamera.exe

# Move core Qt, OpenCV, OpenVINO and ONNX DLLs back to root
# These are the ones that either can't be delay-loaded or are sensitive to search paths
mv build/Release/libs/Qt6*.dll build/Release/ 2>/dev/null
mv build/Release/libs/onnxruntime*.dll build/Release/ 2>/dev/null
mv build/Release/libs/openvino*.dll build/Release/ 2>/dev/null
mv build/Release/libs/opencv_world*.dll build/Release/ 2>/dev/null
mv build/Release/libs/D3DCompiler_47.dll build/Release/ 2>/dev/null
mv build/Release/libs/opengl32sw.dll build/Release/ 2>/dev/null
mv build/Release/libs/vcruntime140*.dll build/Release/ 2>/dev/null
mv build/Release/libs/msvcp140*.dll build/Release/ 2>/dev/null
mv build/Release/libs/tbb*.dll build/Release/ 2>/dev/null

# Clean up any folders that windeployqt might have created in the root despite --dir
# We use rsync to merge directories without 'Directory not empty' errors
for dir in build/Release/*/; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        if [ "$dirname" != "libs" ] && [ "$dirname" != "assets" ]; then
            echo "Merging $dirname into libs/$dirname..."
            mkdir -p "build/Release/libs/$dirname"
            rsync -a "$dir" "build/Release/libs/$dirname/.."
            rm -rf "$dir"
        fi
    fi
done
