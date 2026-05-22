cmake_minimum_required(VERSION 3.16)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Do not override the generator's multi-config settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)
find_package(Qt6 REQUIRED COMPONENTS QuickTest) # Explicitly find QuickTest for this subdirectory

enable_testing()

# Helper function to create a test executable
function(yolo_add_test TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_link_libraries(${TEST_NAME}
        PRIVATE
        gtest_main
        gmock
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        Qt6::Multimedia
        Qt6::Qml
        Qt6::Quick
        Qt6::QuickTest
        ${OpenCV_LIBS}
        openvino::runtime
        onnxruntime
    )
    
    # Include application directories
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/features/inference/domain
        ${CMAKE_SOURCE_DIR}/src/features/inference/application
        ${CMAKE_SOURCE_DIR}/src/features/inference/infrastructure
        ${CMAKE_SOURCE_DIR}/src/features/inference/infrastructure/backends
        ${CMAKE_SOURCE_DIR}/src/features/inference/ui
        ${CMAKE_SOURCE_DIR}/src/features/monitoring/domain
        ${CMAKE_SOURCE_DIR}/src/features/monitoring/application
        ${CMAKE_SOURCE_DIR}/src/features/monitoring/infrastructure
        ${CMAKE_SOURCE_DIR}/src/features/capture/domain
        ${CMAKE_SOURCE_DIR}/src/features/capture/application
        ${CMAKE_SOURCE_DIR}/src/features/capture/infrastructure
        ${CMAKE_SOURCE_DIR}/src/shared/domain
        ${CMAKE_SOURCE_DIR}/src/shared/application
    )

    # Copy DLLs for Windows
    if(WIN32)
        add_custom_command(TARGET ${TEST_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "C:/opencv/build/x64/vc16/bin/opencv_world4120.dll"
                $<TARGET_FILE_DIR:${TEST_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "C:/Qt/6.8.3/msvc2022_64/bin/Qt6Core.dll"
                $<TARGET_FILE_DIR:${TEST_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "C:/Qt/6.8.3/msvc2022_64/bin/Qt6Gui.dll"
                $<TARGET_FILE_DIR:${TEST_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "C:/Qt/6.8.3/msvc2022_64/bin/Qt6Widgets.dll"
                $<TARGET_FILE_DIR:${TEST_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "C:/Qt/6.8.3/msvc2022_64/bin/Qt6Multimedia.dll"
                $<TARGET_FILE_DIR:${TEST_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "C:/Qt/6.8.3/msvc2022_64/bin/Qt6Network.dll"
                $<TARGET_FILE_DIR:${TEST_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "C:/Qt/6.8.3/msvc2022_64/bin/Qt6Qml.dll"
                $<TARGET_FILE_DIR:${TEST_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "C:/Qt/6.8.3/msvc2022_64/bin/Qt6Quick.dll"
                $<TARGET_FILE_DIR:${TEST_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "C:/onnxruntime/lib/onnxruntime.dll"
                $<TARGET_FILE_DIR:${TEST_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "C:/onnxruntime/lib/onnxruntime_providers_shared.dll"
                $<TARGET_FILE_DIR:${TEST_NAME}>
        )
        
        # OpenVINO DLLs (multiple)
        add_custom_command(TARGET ${TEST_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -D "SOURCE_DIR=C:/intel/openvino_toolkit/runtime/bin/intel64/Release"
                    -D "DEST_DIR=$<TARGET_FILE_DIR:${TEST_NAME}>"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/copy_dlls.cmake"
        )
    endif()

    gtest_discover_tests(${TEST_NAME})
endfunction()

# Add PostProcessor Unit Test
yolo_add_test(PostProcessorTest
    unit/features/inference/PostProcessorTest.cpp
    ../src/features/inference/infrastructure/PostProcessor.cpp
    ../src/features/inference/infrastructure/PostProcessor.h
)

# Add PreProcessor Unit Test
yolo_add_test(PreProcessorTest
    unit/features/inference/PreProcessorTest.cpp
    ../src/features/inference/infrastructure/PreProcessor.cpp
    ../src/features/inference/infrastructure/PreProcessor.h
)

# Add InferenceListModel Unit Test
yolo_add_test(InferenceListModelTest
    unit/features/inference/InferenceListModelTest.cpp
    ../src/features/inference/ui/InferenceListModel.cpp
    ../src/features/inference/ui/InferenceListModel.h
    ../src/features/inference/domain/VisionObject.h
)

# Add SystemStats Unit Test
yolo_add_test(SystemStatsTest
    unit/features/monitoring/SystemStatsTest.cpp
    ../src/features/monitoring/domain/SystemStats.h
)

# Add CaptureWorker Unit Test
yolo_add_test(CaptureWorkerTest
    unit/features/capture/CaptureWorkerTest.cpp
    ../src/features/capture/application/CaptureWorker.cpp
    ../src/features/capture/application/CaptureWorker.h
)

# Add SimdUtils Unit Test
yolo_add_test(SimdUtilsTest
    unit/features/inference/SimdUtilsTest.cpp
    ../src/features/inference/infrastructure/SimdUtils.h
)

# Add InferenceWorker Unit Test
yolo_add_test(InferenceWorkerTest
    unit/features/inference/InferenceWorkerTest.cpp
    ../src/features/inference/application/InferenceWorker.cpp
    ../src/features/inference/application/InferenceWorker.h
    ../src/features/inference/domain/TaskType.h
)

# Add InferenceOverlayMath Unit Test
yolo_add_test(InferenceOverlayMathTest
    unit/features/inference/InferenceOverlayMathTest.cpp
    ../src/features/inference/ui/RenderTransform.h
)

# Add OpenCVImageFileSource Unit Test
yolo_add_test(OpenCVImageFileSourceTest
    unit/features/capture/OpenCVImageFileSourceTest.cpp
    ../src/features/capture/infrastructure/OpenCVImageFileSource.cpp
)

# Add FFmpegVideoSource Unit Test
yolo_add_test(FFmpegVideoSourceTest
    unit/features/capture/FFmpegVideoSourceTest.cpp
    ../src/features/capture/infrastructure/FFmpegVideoSource.cpp
)
target_link_libraries(FFmpegVideoSourceTest PRIVATE avcodec avformat avutil swscale)

# Add AppController Unit Test
yolo_add_test(AppControllerTest
    unit/shared/AppControllerTest.cpp
    ../src/shared/application/AppController.cpp
    ../src/shared/application/AppController.h
    ../src/features/monitoring/application/MonitoringController.cpp
    ../src/features/monitoring/application/MonitoringController.h
    ../src/features/monitoring/application/SystemMonitorWorker.cpp
    ../src/features/monitoring/application/SystemMonitorWorker.h
    ../src/features/monitoring/infrastructure/WindowsSystemMonitor.cpp
    ../src/features/monitoring/infrastructure/WindowsSystemMonitor.h
    ../src/features/inference/application/InferenceController.cpp
    ../src/features/inference/application/InferenceController.h
    ../src/features/inference/application/InferenceWorker.cpp
    ../src/features/inference/application/InferenceWorker.h
    ../src/features/capture/application/CaptureWorker.cpp
    ../src/features/capture/application/CaptureWorker.h
    ../src/features/capture/application/YoloCameraController.cpp
    ../src/features/capture/application/YoloCameraController.h
    ../src/features/capture/application/VideoFileController.cpp
    ../src/features/capture/application/VideoFileController.h
    ../src/features/capture/application/ImageFileController.cpp
    ../src/features/capture/application/ImageFileController.h
    ../src/features/inference/ui/InferenceListModel.cpp
    ../src/features/inference/ui/InferenceListModel.h
    ../src/features/inference/ui/RenderTransform.h
    ../src/features/inference/domain/VisionObject.h
    ../src/features/inference/domain/TaskType.h
    ../src/features/inference/infrastructure/YoloPipeline.cpp
    ../src/features/inference/infrastructure/PreProcessor.cpp
    ../src/features/inference/infrastructure/PostProcessor.cpp
    ../src/features/inference/infrastructure/backends/OnnxRuntimeBackend.cpp
    ../src/features/inference/infrastructure/backends/OpenVinoBackend.cpp
    ../src/features/capture/infrastructure/OpenCVCameraSource.cpp
    ../src/features/capture/infrastructure/OpenCVImageFileSource.cpp
    ../src/features/capture/infrastructure/FFmpegVideoSource.cpp
)
target_link_libraries(AppControllerTest PRIVATE avcodec avformat avutil swscale pdh psapi)

# Add WindowsSystemMonitor Unit Test
yolo_add_test(WindowsSystemMonitorTest
    unit/features/monitoring/WindowsSystemMonitorTest.cpp
    ../src/features/monitoring/infrastructure/WindowsSystemMonitor.cpp
)
target_link_libraries(WindowsSystemMonitorTest PRIVATE pdh psapi)

# Add YoloPipeline Unit Test
# Note: YoloPipeline has many dependencies, we mostly test its setup/logic
yolo_add_test(YoloPipelineTest
    unit/features/inference/YoloPipelineTest.cpp
    ../src/features/inference/infrastructure/YoloPipeline.cpp
    ../src/features/inference/infrastructure/YoloPipeline.h
    ../src/features/inference/infrastructure/PreProcessor.cpp
    ../src/features/inference/infrastructure/PostProcessor.cpp
    ../src/features/inference/infrastructure/backends/OnnxRuntimeBackend.cpp
    ../src/features/inference/infrastructure/backends/OpenVinoBackend.cpp
    ../src/features/inference/domain/TaskType.h
)

# Add QML Tests
add_executable(QMLTests 
    qml/qmltestrunner.cpp
    ../src/features/inference/ui/InferenceOverlayItem.cpp
    ../src/features/inference/ui/InferenceListModel.cpp
    ../src/features/inference/domain/VisionObject.h
    ../src/features/inference/domain/TaskType.h
)
target_compile_definitions(QMLTests PRIVATE 
    QUICK_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/qml"
)
target_link_libraries(QMLTests PRIVATE
    gtest_main
    gmock
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Multimedia
    Qt6::Qml
    Qt6::Quick
    Qt6::QuickTest
    ${OpenCV_LIBS}
)
target_include_directories(QMLTests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/features/inference/ui
    ${CMAKE_SOURCE_DIR}/src/features/inference/domain
)

if(WIN32)
    add_custom_command(TARGET QMLTests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/Qt/6.8.3/msvc2022_64/bin/Qt6Core.dll"
            $<TARGET_FILE_DIR:QMLTests>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/Qt/6.8.3/msvc2022_64/bin/Qt6Gui.dll"
            $<TARGET_FILE_DIR:QMLTests>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/Qt/6.8.3/msvc2022_64/bin/Qt6Widgets.dll"
            $<TARGET_FILE_DIR:QMLTests>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/Qt/6.8.3/msvc2022_64/bin/Qt6Qml.dll"
            $<TARGET_FILE_DIR:QMLTests>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/Qt/6.8.3/msvc2022_64/bin/Qt6Quick.dll"
            $<TARGET_FILE_DIR:QMLTests>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/Qt/6.8.3/msvc2022_64/bin/Qt6Multimedia.dll"
            $<TARGET_FILE_DIR:QMLTests>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/Qt/6.8.3/msvc2022_64/bin/Qt6QuickTest.dll"
            $<TARGET_FILE_DIR:QMLTests>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/Qt/6.8.3/msvc2022_64/bin/Qt6Network.dll"
            $<TARGET_FILE_DIR:QMLTests>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/Qt/6.8.3/msvc2022_64/bin/Qt6OpenGL.dll"
            $<TARGET_FILE_DIR:QMLTests>
    )
endif()

# We need to manually register the test with ctest
add_test(NAME QMLTests COMMAND QMLTests 
    -import ${CMAKE_SOURCE_DIR}/src/ui 
    -import ${CMAKE_BINARY_DIR}
    -import ${CMAKE_BINARY_DIR}/Release/qml
)


