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
find_package(Qt6 REQUIRED COMPONENTS Gui Widgets Quick Multimedia QuickControls2 QuickDialogs2 QuickTest)
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

# 5. Setup FFmpeg
set(FFMPEG_DIR "C:/ffmpeg")
include_directories("${FFMPEG_DIR}/include")
link_directories("${FFMPEG_DIR}/lib")

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

    # ── Inference Feature ──
    src/features/inference/domain/TaskType.h
    src/features/inference/domain/InferenceResult.h
    src/features/inference/domain/InferenceConfig.h
    src/features/inference/domain/InferenceTiming.h
    src/features/inference/domain/IInferenceModel.h
    src/features/inference/infrastructure/SimdUtils.h
    src/features/inference/infrastructure/PreProcessor.h
    src/features/inference/infrastructure/PreProcessor.cpp
    src/features/inference/infrastructure/PostProcessor.h
    src/features/inference/infrastructure/PostProcessor.cpp
    src/features/inference/infrastructure/YoloPipeline.h
    src/features/inference/infrastructure/YoloPipeline.cpp
    src/features/inference/infrastructure/backends/IInferenceBackend.h
    src/features/inference/infrastructure/backends/OnnxRuntimeBackend.h
    src/features/inference/infrastructure/backends/OnnxRuntimeBackend.cpp
    src/features/inference/infrastructure/backends/OpenVinoBackend.h
    src/features/inference/infrastructure/backends/OpenVinoBackend.cpp
    src/features/inference/domain/VisionObject.h
    src/features/inference/application/InferenceWorker.h
    src/features/inference/application/InferenceWorker.cpp
    src/features/inference/application/InferenceController.h
    src/features/inference/application/InferenceController.cpp
    src/features/inference/ui/InferenceListModel.h
    src/features/inference/ui/InferenceListModel.cpp
    src/features/inference/ui/InferenceOverlayItem.h
    src/features/inference/ui/InferenceOverlayItem.cpp
    src/features/inference/ui/RenderTransform.h

    # ── Capture Feature ──
    src/features/capture/domain/SourceConfig.h
    src/features/capture/domain/CameraFrame.h
    src/features/capture/domain/ICaptureSource.h
    src/features/capture/infrastructure/OpenCVCameraSource.h
    src/features/capture/infrastructure/OpenCVCameraSource.cpp
    src/features/capture/infrastructure/OpenCVImageFileSource.h
    src/features/capture/infrastructure/OpenCVImageFileSource.cpp
    src/features/capture/infrastructure/FFmpegVideoSource.h
    src/features/capture/infrastructure/FFmpegVideoSource.cpp
    src/features/capture/application/CaptureWorker.h
    src/features/capture/application/CaptureWorker.cpp
    src/features/capture/application/YoloCameraController.h
    src/features/capture/application/YoloCameraController.cpp
    src/features/capture/application/VideoFileController.h
    src/features/capture/application/VideoFileController.cpp
    src/features/capture/application/ImageFileController.h
    src/features/capture/application/ImageFileController.cpp
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
    src/ui/Main.qml
    src/ui/AppHeader.qml
    src/ui/PlaybackControls.qml
    src/ui/PerformancePanel.qml
    src/ui/YoloOverlay.qml
    src/ui/CustomComboBox.qml
    src/ui/MetricItem.qml
)

# 7. Include Directories (Clean Architecture approach)
target_include_directories(appCamera PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    src/features/inference/domain
    src/features/inference/application
    src/features/inference/infrastructure
    src/features/inference/ui
    src/features/monitoring/domain
    src/features/monitoring/application
    src/features/monitoring/infrastructure
    src/features/capture/domain
    src/features/capture/application
    src/features/capture/infrastructure
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
    avcodec
    avformat
    avutil
    swscale
)

# Silence debug output in Release builds
# target_compile_definitions(appCamera PRIVATE
#     $<$<CONFIG:Release>:QT_NO_DEBUG_OUTPUT>
# )

if(WIN32)
    target_link_libraries(appCamera PRIVATE pdh psapi)
endif()

# 10. Testing
option(BUILD_TESTING "Build the testing suite" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# 11. Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/../docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
endif()

# 9. Post-build: DLL Deployment
file(GLOB OPENVINO_DLLS "C:/intel/openvino_toolkit/runtime/bin/intel64/Release/*.dll")
file(GLOB TBB_DLLS "C:/intel/openvino_toolkit/runtime/3rdparty/tbb/bin/*.dll")

add_custom_command(TARGET appCamera POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "[Post-Build] Deploying assets to execution directory..."
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:appCamera>/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/../assets
        $<TARGET_FILE_DIR:appCamera>/assets
)
