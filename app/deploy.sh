#!/bin/bash
set -e

RELEASE_DIR="build/Release"
LIBS_DIR="$RELEASE_DIR/libs"

# 1. Deploy Qt dependencies into libs/
C:/Qt/6.8.3/msvc2022_64/bin/windeployqt.exe --release --qmldir content --qmldir . --dir "$LIBS_DIR" "$RELEASE_DIR/appCamera.exe"

# 2. Organize Qt DLLs into libs/qt/
mkdir -p "$LIBS_DIR/qt"
mv "$LIBS_DIR"/Qt6*.dll "$LIBS_DIR/qt/" 2>/dev/null
mv "$LIBS_DIR"/D3DCompiler_47.dll "$LIBS_DIR/qt/" 2>/dev/null
mv "$LIBS_DIR"/opengl32sw.dll "$LIBS_DIR/qt/" 2>/dev/null
mv "$LIBS_DIR"/dxcompiler.dll "$LIBS_DIR/qt/" 2>/dev/null

# 3. Organize FFmpeg DLLs into libs/ffmpeg/
mkdir -p "$LIBS_DIR/ffmpeg"
mv "$LIBS_DIR"/av*.dll "$LIBS_DIR/ffmpeg/" 2>/dev/null
mv "$LIBS_DIR"/sw*.dll "$LIBS_DIR/ffmpeg/" 2>/dev/null

# 4. Move Qt plugins into libs/plugins/ (windeployqt creates subdirs like generic/, platforms/, etc.)
mkdir -p "$LIBS_DIR/plugins"
for plugindir in generic iconengines imageformats multimedia networkinformation platforms tls; do
    if [ -d "$LIBS_DIR/$plugindir" ]; then
        cp -r "$LIBS_DIR/$plugindir" "$LIBS_DIR/plugins/"
        rm -rf "$LIBS_DIR/$plugindir"
    fi
done

# 5. Clean up any stray folders from the root (not libs/ or assets/)
for dir in "$RELEASE_DIR"/*/; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        if [ "$dirname" != "libs" ] && [ "$dirname" != "assets" ]; then
            echo "Merging $dirname into libs/$dirname..."
            cp -r "$dir" "$LIBS_DIR/"
            rm -rf "$dir"
        fi
    fi
done

# 6. Remove debug DLLs from release deployment
rm -f "$LIBS_DIR/openvino/"*_debug.dll 2>/dev/null
rm -f "$LIBS_DIR/opencv/opencv_world4120d.dll" 2>/dev/null

# 7. Remove build artifacts
rm -f "$RELEASE_DIR"/appCamera.lib "$RELEASE_DIR"/appCamera.exp 2>/dev/null

# 8. Copy onnxruntime DLLs to root so Windows finds them before C:\Windows\System32
cp "$LIBS_DIR/onnx/onnxruntime.dll" "$RELEASE_DIR/" 2>/dev/null
cp "$LIBS_DIR/onnx/onnxruntime_providers_shared.dll" "$RELEASE_DIR/" 2>/dev/null

# 9. Create launcher script that sets PATH for categorized DLL dirs
cat > "$RELEASE_DIR/appCamera.bat" << 'LAUNCHER'
@echo off
setlocal
set "APP_DIR=%~dp0"
set "PATH=%APP_DIR%libs\qt;%APP_DIR%libs\opencv;%APP_DIR%libs\openvino;%APP_DIR%libs\onnx;%APP_DIR%libs\ffmpeg;%PATH%"
start "" "%APP_DIR%appCamera.exe" %*
LAUNCHER

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Root contents:"
ls -F "$RELEASE_DIR"
echo ""
echo "libs/ contents:"
ls -F "$LIBS_DIR"
