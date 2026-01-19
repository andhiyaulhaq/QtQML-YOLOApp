cmake_minimum_required(VERSION 3.16)

project(QtOpenCVCamera VERSION 0.1 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 1. Setup Qt6
find_package(Qt6 REQUIRED COMPONENTS Quick)
qt_standard_project_setup()
qt_policy(SET QTP0001 NEW)

# 2. Setup OpenCV (Adjust path if necessary)
set(OpenCV_DIR "C:/opencv/build_mingw")
find_package(OpenCV REQUIRED)

# 3. Create Executable
qt_add_executable(appCamera
    main.cpp
    src/OpenCVItem.cpp
    src/OpenCVItem.h
)

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
    ${OpenCV_LIBS}
)
