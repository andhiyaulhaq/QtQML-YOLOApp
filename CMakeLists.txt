cmake_minimum_required(VERSION 3.16)

project(QtOpenCVCamera VERSION 0.1 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 1. Setup Qt6 (Added Multimedia component)
find_package(Qt6 REQUIRED COMPONENTS Quick Multimedia)

qt_standard_project_setup()
qt_policy(SET QTP0001 NEW)

# 2. Setup OpenCV
set(OpenCV_DIR "C:/opencv/build_mingw")
find_package(OpenCV REQUIRED)

# 3. Setup ONNX Runtime
include_directories("C:/onnxruntime/include")
link_directories("C:/onnxruntime/lib")

# 3. Create Executable (Updated to use VideoController)
qt_add_executable(appCamera
    main.cpp
    src/VideoController.cpp
    src/VideoController.h
    src/BoundingBoxItem.cpp
    src/BoundingBoxItem.h
    src/inference.cpp
    src/inference.h
    src/SystemMonitor.cpp
    src/SystemMonitor.h
    src/DetectionStruct.h
    src/DetectionListModel.cpp
    src/DetectionListModel.h
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
)

# 6. Link Libraries
target_link_libraries(appCamera
    PRIVATE
    Qt6::Quick
    Qt6::Multimedia
    ${OpenCV_LIBS}
    onnxruntime
    onnxruntime_providers_shared
)

# 7. Platform-specific libraries for system monitoring
if(WIN32)
    target_link_libraries(appCamera PRIVATE pdh psapi)
endif()

# 8. Copy ONNX Runtime DLLs to build directory
add_custom_command(TARGET appCamera POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        C:/onnxruntime/lib/onnxruntime.dll
        C:/onnxruntime/lib/onnxruntime_providers_shared.dll
        $<TARGET_FILE_DIR:appCamera>
)

# 9. Copy model files to build directory
add_custom_command(TARGET appCamera POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/inference
        $<TARGET_FILE_DIR:appCamera>/inference
)
