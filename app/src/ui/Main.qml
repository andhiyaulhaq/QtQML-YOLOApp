import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import QtMultimedia 6.0
import CameraModule 1.0

Window {
    id: root
    width: 1400
    height: 800
    visible: true
    title: "YOLOApp"
    color: "#121212"

    property string inputMode: "image" // "camera", "video", "image"

    FileDialog {
        id: videoFileDialog
        title: "Select Video File"
        nameFilters: ["Video files (*.mp4 *.avi *.mkv *.mov *.wmv)"]
        onAccepted: {
            root.inputMode = "video"
            videoFile.setFilePath(selectedFile)
        }
        onRejected: {
            if (root.inputMode === "video" && !videoFile.hasFile) {
                sourceCombo.currentIndex = 0
                root.inputMode = "camera"
                camera.activate()
            }
        }
    }

    FileDialog {
        id: imageFileDialog
        title: "Select Image File"
        nameFilters: ["Image files (*.jpg *.jpeg *.png *.bmp)"]
        onAccepted: {
            root.inputMode = "image"
            imageFile.setFilePath(selectedFile)
        }
        onRejected: {
            if (root.inputMode === "image" && !imageFile.hasFile) {
                sourceCombo.currentIndex = 0
                root.inputMode = "camera"
                camera.activate()
            }
        }
    }


    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 15

        AppHeader {
            inputMode: root.inputMode
            cameraSource: camera
            videoSource: videoFile
            imageSource: imageFile
            inferenceController: inference
            
            onSourceChanged: (index) => {
                if (index === 0) {
                    root.inputMode = "camera"
                    camera.activate()
                } else if (index === 1) {
                    if (videoFile.hasFile) {
                        root.inputMode = "video"
                        videoFile.activate()
                    } else {
                        videoFileDialog.open()
                    }
                } else if (index === 2) {
                    if (imageFile.hasFile) {
                        root.inputMode = "image"
                        imageFile.activate()
                    } else {
                        imageFileDialog.open()
                    }
                }
            }
            
            onBrowseRequested: {
                if (inputMode === "video") videoFileDialog.open();
                else if (inputMode === "image") imageFileDialog.open();
            }
            
            onTaskChanged: (index) => {
                if (!inference) return;
                if (index === 0) inference.currentTask = YoloTask.ObjectDetection
                else if (index === 1) inference.currentTask = YoloTask.PoseEstimation
                else if (index === 2) inference.currentTask = YoloTask.ImageSegmentation
            }
            
            onRuntimeChanged: (index) => {
                if (!inference) return;
                if (index === 0) inference.currentRuntime = YoloTask.OpenVINO
                else if (index === 1) inference.currentRuntime = YoloTask.ONNXRuntime
            }
            
            onResChanged: (index) => {
                if (camera) {
                    var res = camera.supportedResolutions[index]
                    camera.currentResolution = res
                }
            }
        }

        // Main Layout
        RowLayout {
            spacing: 20
            Layout.fillWidth: true
            Layout.fillHeight: true

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: "black"
                radius: 8
                clip: true

                VideoOutput {
                    id: videoOutput
                    anchors.fill: parent
                    fillMode: VideoOutput.PreserveAspectFit
                    Component.onCompleted: {
                        if (camera) camera.videoSink = videoOutput.videoSink
                    }
                }

                YoloOverlay {
                    anchors.fill: videoOutput
                    inferenceController: inference
                }

                // Play/pause mouse area and overlay
                MouseArea {
                    id: videoMouseArea
                    anchors.fill: videoOutput
                    anchors.bottomMargin: playbackControls.visible ? playbackControls.height : 0
                    visible: root.inputMode === "video"
                    
                    onClicked: {
                        if (typeof videoFile !== "undefined" && videoFile) {
                            videoFile.togglePlayPause()
                        }
                    }
                }

                // Play icon overlay
                Rectangle {
                    id: playOverlay
                    anchors.fill: videoOutput
                    anchors.bottomMargin: playbackControls.visible ? playbackControls.height : 0
                    color: "#30000000" // subtle dimming
                    visible: root.inputMode === "video" && (typeof videoFile !== "undefined" && videoFile && videoFile.isPaused)
                    
                    MouseArea {
                        anchors.fill: parent
                        onClicked: {
                            if (typeof videoFile !== "undefined" && videoFile) {
                                videoFile.togglePlayPause()
                            }
                        }
                    }

                    Rectangle {
                        anchors.centerIn: parent
                        width: 90
                        height: 90
                        radius: 45
                        color: "#B3FFFFFF"
                        border.color: "#80FFFFFF"
                        border.width: 1

                        MouseArea {
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: {
                                if (typeof videoFile !== "undefined" && videoFile) {
                                    videoFile.togglePlayPause()
                                }
                            }
                        }

                        Text {
                            anchors.centerIn: parent
                            anchors.horizontalCenterOffset: 5
                            text: "▶"
                            color: "#121212"
                            font.pixelSize: 36
                        }
                    }
                }

                PlaybackControls {
                    id: playbackControls
                    anchors.bottom: parent.bottom
                    anchors.left: parent.left
                    anchors.right: parent.right
                    visible: root.inputMode === "video"
                    videoSource: videoFile
                }
            }

            PerformancePanel {
                Layout.fillHeight: true
                inputMode: root.inputMode
                cameraSource: camera
                inferenceController: inference
                monitoringSource: monitoring
            }
        }
    }
}
