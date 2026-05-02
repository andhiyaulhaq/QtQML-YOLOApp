#!/bin/bash
set -e

# Project configuration
RELEASE_DIR="build/Release"
LIBS_DIR="$RELEASE_DIR/libs"

echo "------------------------------------------------"
echo "  YOLOApp: Deployment & Packaging"
echo "------------------------------------------------"

# 1. Deploy Qt dependencies into libs/
echo "[1/7] Running windeployqt..."
C:/Qt/6.8.3/msvc2022_64/bin/windeployqt.exe --release --qmldir content --dir "$LIBS_DIR" "$RELEASE_DIR/appCamera.exe"

# 2. Organize Qt DLLs into libs/qt/
echo "[2/7] Organizing Qt libraries..."
mkdir -p "$LIBS_DIR/qt"
mv "$LIBS_DIR"/Qt6*.dll "$LIBS_DIR/qt/" 2>/dev/null
mv "$LIBS_DIR"/D3DCompiler_47.dll "$LIBS_DIR/qt/" 2>/dev/null
mv "$LIBS_DIR"/opengl32sw.dll "$LIBS_DIR/qt/" 2>/dev/null
mv "$LIBS_DIR"/dxcompiler.dll "$LIBS_DIR/qt/" 2>/dev/null

# 3. Organize Third-Party DLLs (OpenCV, OpenVINO, ONNX)
echo "[3/7] Organizing third-party libraries..."

# OpenCV
mkdir -p "$LIBS_DIR/opencv"
cp "C:/opencv/build/x64/vc16/bin/opencv_world4120.dll" "$LIBS_DIR/opencv/" 2>/dev/null

# OpenVINO
mkdir -p "$LIBS_DIR/openvino"
cp C:/intel/openvino_toolkit/runtime/bin/intel64/Release/*.dll "$LIBS_DIR/openvino/" 2>/dev/null
cp C:/intel/openvino_toolkit/runtime/3rdparty/tbb/bin/*.dll "$LIBS_DIR/openvino/" 2>/dev/null
rm -f "$LIBS_DIR/openvino/"*_debug.dll 2>/dev/null

# ONNX Runtime
mkdir -p "$LIBS_DIR/onnx"
cp "C:/onnxruntime/lib/onnxruntime.dll" "$LIBS_DIR/onnx/" 2>/dev/null
cp "C:/onnxruntime/lib/onnxruntime_providers_shared.dll" "$LIBS_DIR/onnx/" 2>/dev/null
# Keep ONNX DLLs in root as well for easier lookup if needed, but per legacy we set PATH
cp "$LIBS_DIR/onnx/onnxruntime.dll" "$RELEASE_DIR/" 2>/dev/null
cp "$LIBS_DIR/onnx/onnxruntime_providers_shared.dll" "$RELEASE_DIR/" 2>/dev/null

# 4. Organize Qt plugins into libs/plugins/
echo "[4/7] Organizing Qt plugins..."
mkdir -p "$LIBS_DIR/plugins"
for plugindir in generic iconengines imageformats multimedia networkinformation platforms tls; do
    if [ -d "$LIBS_DIR/$plugindir" ]; then
        cp -r "$LIBS_DIR/$plugindir" "$LIBS_DIR/plugins/"
        rm -rf "$LIBS_DIR/$plugindir"
    fi
done

# 5. Deploy Assets
echo "[5/7] Deploying assets..."
mkdir -p "$RELEASE_DIR/assets"
cp -r ../assets/* "$RELEASE_DIR/assets/" 2>/dev/null

# 6. Create Launcher Script (appCamera.bat)
echo "[6/7] Creating launcher script..."
cat > "$RELEASE_DIR/appCamera.bat" << 'LAUNCHER'
@echo off
setlocal
set "APP_DIR=%~dp0"
set "PATH=%APP_DIR%libs\qt;%APP_DIR%libs\opencv;%APP_DIR%libs\openvino;%APP_DIR%libs\onnx;%PATH%"
set "QT_QPA_PLATFORM_PLUGIN_PATH=%APP_DIR%libs\plugins\platforms"
set "QT_PLUGIN_PATH=%APP_DIR%libs\plugins"
set "QML2_IMPORT_PATH=%APP_DIR%libs\qml"
"%APP_DIR%appCamera.exe" %*
LAUNCHER

# 7. Cleanup
echo "[7/7] Cleaning up..."
rm -f "$RELEASE_DIR"/appCamera.lib "$RELEASE_DIR"/appCamera.exp 2>/dev/null

echo "------------------------------------------------"
echo "  Deployment Complete!"
echo "  Run via: $RELEASE_DIR/appCamera.bat"
echo "------------------------------------------------"
