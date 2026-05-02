cmake_minimum_required(VERSION 3.16)

project(QtOpenCVCamera VERSION 0.1 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(/Zc:__cplusplus /permissive- /utf-8)
    add_definitions(-DNOMINMAX)
endif()

# 1. Setup Qt6
find_package(Qt6 REQUIRED COMPONENTS Quick Multimedia QuickControls2 QuickDialogs2)
qt_standard_project_setup()

# 2. Setup OpenCV
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)

# 3. Setup OpenVINO
set(OpenVINO_DIR "C:/intel/openvino_toolkit/runtime/cmake")
find_package(OpenVINO REQUIRED)

# 4. Setup ONNX Runtime
include_directories("C:/onnxruntime/include")
link_directories("C:/onnxruntime/lib")

# 5. Create Executable
qt_policy(SET QTP0001 NEW)
qt_policy(SET QTP0004 NEW)

qt_add_executable(appCamera
    src/main.cpp
    src/shared/domain/AppConfig.h
    
    # ── Shared Application (Composition Root) ──
    src/shared/application/AppController.h
    src/shared/application/AppController.cpp
    src/shared/domain/UiLogger.h

    # ── Monitoring Feature ──
    src/features/monitoring/domain/SystemStats.h
    src/features/monitoring/domain/ISystemMonitor.h
    src/features/monitoring/application/SystemMonitorWorker.h
    src/features/monitoring/application/SystemMonitorWorker.cpp
    src/features/monitoring/application/MonitoringController.h
    src/features/monitoring/application/MonitoringController.cpp
    src/features/monitoring/infrastructure/WindowsSystemMonitor.h
    src/features/monitoring/infrastructure/WindowsSystemMonitor.cpp

    # ── Detection Feature ──
    src/features/detection/domain/TaskType.h
    src/features/detection/domain/DetectionResult.h
    src/features/detection/domain/InferenceConfig.h
    src/features/detection/domain/InferenceTiming.h
    src/features/detection/domain/IDetectionModel.h
    src/features/detection/infrastructure/SimdUtils.h
    src/features/detection/infrastructure/PreProcessor.h
    src/features/detection/infrastructure/PreProcessor.cpp
    src/features/detection/infrastructure/PostProcessor.h
    src/features/detection/infrastructure/PostProcessor.cpp
    src/features/detection/infrastructure/YoloPipeline.h
    src/features/detection/infrastructure/YoloPipeline.cpp
    src/features/detection/infrastructure/backends/IInferenceBackend.h
    src/features/detection/infrastructure/backends/OnnxRuntimeBackend.h
    src/features/detection/infrastructure/backends/OnnxRuntimeBackend.cpp
    src/features/detection/infrastructure/backends/OpenVinoBackend.h
    src/features/detection/infrastructure/backends/OpenVinoBackend.cpp
    src/features/detection/domain/Detection.h
    src/features/detection/application/InferenceWorker.h
    src/features/detection/application/InferenceWorker.cpp
    src/features/detection/application/DetectionController.h
    src/features/detection/application/DetectionController.cpp
    src/features/detection/ui/DetectionListModel.h
    src/features/detection/ui/DetectionListModel.cpp
    src/features/detection/ui/DetectionOverlayItem.h
    src/features/detection/ui/DetectionOverlayItem.cpp

    # ── Camera Feature ──
    src/features/camera/domain/SourceConfig.h
    src/features/camera/domain/CameraFrame.h
    src/features/camera/domain/ICaptureSource.h
    src/features/camera/infrastructure/OpenCVCameraSource.h
    src/features/camera/infrastructure/OpenCVCameraSource.cpp
    src/features/camera/infrastructure/OpenCVVideoFileSource.h
    src/features/camera/infrastructure/OpenCVVideoFileSource.cpp
    src/features/camera/application/CaptureWorker.h
    src/features/camera/application/CaptureWorker.cpp
    src/features/camera/application/YoloCameraController.h
    src/features/camera/application/YoloCameraController.cpp
    src/features/camera/application/VideoFileController.h
    src/features/camera/application/VideoFileController.cpp
)

# Optimization Flags
if(MSVC)
    target_compile_options(appCamera PRIVATE /O2 /Ob2 /Oi /Ot /Oy /GL)
    target_link_options(appCamera PRIVATE /LTCG)
else()
    target_compile_options(appCamera PRIVATE -O3 -march=native -ffast-math)
endif()

# 6. Register QML Module
qt_add_qml_module(appCamera
    URI "CameraModule"
    VERSION 1.0
    QML_FILES
    content/Main.qml
)

# 7. Include Directories (Clean Architecture approach)
target_include_directories(appCamera PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    src/features/detection/domain
    src/features/detection/application
    src/features/detection/infrastructure
    src/features/detection/ui
    src/features/monitoring/domain
    src/features/monitoring/application
    src/features/monitoring/infrastructure
    src/features/camera/domain
    src/features/camera/application
    src/features/camera/infrastructure
    src/shared/domain
    src/shared/application
)

# 8. Link Libraries
target_link_libraries(appCamera
    PRIVATE
    Qt6::Quick
    Qt6::QuickControls2
    Qt6::QuickDialogs2
    Qt6::Multimedia
    ${OpenCV_LIBS}
    openvino::runtime
    onnxruntime
    onnxruntime_providers_shared
)

# Silence debug output in Release builds
# target_compile_definitions(appCamera PRIVATE
#     $<$<CONFIG:Release>:QT_NO_DEBUG_OUTPUT>
# )

if(WIN32)
    target_link_libraries(appCamera PRIVATE pdh psapi)
endif()

# 9. Post-build: DLL Deployment
file(GLOB OPENVINO_DLLS "C:/intel/openvino_toolkit/runtime/bin/intel64/Release/*.dll")
file(GLOB TBB_DLLS "C:/intel/openvino_toolkit/runtime/3rdparty/tbb/bin/*.dll")

add_custom_command(TARGET appCamera POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:appCamera>/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/../assets
        $<TARGET_FILE_DIR:appCamera>/assets
    COMMENT "Deploying assets..."
)
