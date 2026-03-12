cmake_minimum_required(VERSION 3.16)

project(QtOpenCVCamera VERSION 0.1 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(/Zc:__cplusplus /permissive- /utf-8)
    add_definitions(-DNOMINMAX)
endif()

# 1. Setup Qt6 (Added Multimedia component)
find_package(Qt6 REQUIRED COMPONENTS Quick Multimedia QuickControls2)

qt_standard_project_setup()
qt_policy(SET QTP0001 NEW)

# 2. Setup OpenCV
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)

# 3. Setup OpenVINO
set(OpenVINO_DIR "C:/intel/openvino_toolkit/runtime/cmake")
find_package(OpenVINO REQUIRED)

# 4. Setup ONNX Runtime
include_directories("C:/onnxruntime/include")
link_directories("C:/onnxruntime/lib")

# 3. Create Executable (Updated to use VideoController)
qt_add_executable(appCamera
    main.cpp
    src/core/VideoController.cpp
    src/core/VideoController.h
    src/ui/DetectionOverlayItem.cpp
    src/ui/DetectionOverlayItem.h
    src/pipeline/YoloPipeline.cpp
    src/pipeline/PreProcessor.cpp
    src/pipeline/PostProcessor.cpp
    src/pipeline/backends/OnnxRuntimeBackend.cpp
    src/pipeline/backends/OpenVinoBackend.cpp
    src/pipeline/YoloPipeline.h
    src/pipeline/backends/IInferenceBackend.h
    src/pipeline/backends/OnnxRuntimeBackend.h
    src/pipeline/backends/OpenVinoBackend.h
    src/pipeline/YoloTypes.h
    src/pipeline/SimdUtils.h
    src/core/SystemMonitor.cpp
    src/core/SystemMonitor.h
    src/models/DetectionStruct.h
    src/models/DetectionListModel.cpp
    src/models/DetectionListModel.h
)

# Optimization Flags for Release
if(MSVC)
    target_compile_options(appCamera PRIVATE /O2 /Ob2 /Oi /Ot /Oy /GL)
    target_link_options(appCamera PRIVATE /LTCG)
else()
    target_compile_options(appCamera PRIVATE -O3 -march=native -ffast-math)
endif()

# target_compile_options(appCamera PRIVATE -Wno-macro-redefined)

# 4. Register QML Module
qt_add_qml_module(appCamera
    URI "CameraModule"
    VERSION 1.0
    QML_FILES
        content/Main.qml
)

# 5. Include Directories
target_include_directories(appCamera PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/ui
    ${CMAKE_SOURCE_DIR}/src/models
    ${CMAKE_SOURCE_DIR}/src/pipeline
)

# 6. Link Libraries
target_link_libraries(appCamera
    PRIVATE
    Qt6::Quick
    Qt6::QuickControls2
    Qt6::Multimedia
    ${OpenCV_LIBS}
    openvino::runtime
    onnxruntime
    onnxruntime_providers_shared
)

# 7. Platform-specific libraries for system monitoring
if(WIN32)
    target_link_libraries(appCamera PRIVATE pdh psapi)
endif()

# 8. Copy Runtime DLLs to build directory
file(GLOB OPENCV_DLLS "C:/opencv/build/x64/vc16/bin/*.dll")
file(GLOB OPENVINO_DLLS "C:/intel/openvino_toolkit/runtime/bin/intel64/Release/*.dll")
file(GLOB TBB_DLLS "C:/intel/openvino_toolkit/runtime/3rdparty/tbb/bin/*.dll")

add_custom_command(TARGET appCamera POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        C:/onnxruntime/lib/onnxruntime.dll
        C:/onnxruntime/lib/onnxruntime_providers_shared.dll
        ${OPENCV_DLLS}
        ${OPENVINO_DLLS}
        ${TBB_DLLS}
        $<TARGET_FILE_DIR:appCamera>
    COMMENT "Copying Runtime DLLs to the build directory..."
)

# 9. Copy model files to build directory
add_custom_command(TARGET appCamera POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets
        $<TARGET_FILE_DIR:appCamera>/assets
)
