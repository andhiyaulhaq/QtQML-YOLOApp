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
            detectionController: detection
            
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
                if (!detection) return;
                if (index === 0) detection.currentTask = YoloTask.ObjectDetection
                else if (index === 1) detection.currentTask = YoloTask.PoseEstimation
                else if (index === 2) detection.currentTask = YoloTask.ImageSegmentation
            }
            
            onRuntimeChanged: (index) => {
                if (!detection) return;
                if (index === 0) detection.currentRuntime = YoloTask.OpenVINO
                else if (index === 1) detection.currentRuntime = YoloTask.ONNXRuntime
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
                    detectionController: detection
                }

                PlaybackControls {
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
                detectionController: detection
                monitoringSource: monitoring
            }
        }

        // Footer
        Button {
            text: "Exit Application"
            Layout.alignment: Qt.AlignHCenter
            onClicked: Qt.quit()
            
            contentItem: Text {
                text: parent.text
                color: parent.pressed ? "#FF5252" : "#FFFFFF"
                font.bold: true
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
            
            background: Rectangle {
                implicitWidth: 150
                implicitHeight: 40
                color: parent.hovered ? "#333333" : "#222222"
                radius: 4
                border.color: "#444444"
                border.width: 1
            }
        }
    }
}
