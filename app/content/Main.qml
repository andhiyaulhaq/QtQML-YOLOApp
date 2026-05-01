import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtMultimedia 6.0
import CameraModule 1.0

Window {
    width: 1000
    height: 700
    visible: true
    title: "YOLO Object Detection - Metrics Dashboard"
    color: "#121212"

    VideoController {
        id: controller
        videoSink: videoOutput.videoSink
        onErrorOccurred: (title, message) => {
            errorDialog.title = title
            errorText.text = message
            errorDialog.open()
        }
    }

    Dialog {
        id: errorDialog
        anchors.centerIn: parent
        width: 400
        modal: true
        title: "Error"
        
        background: Rectangle {
            color: "#1e1e1e"
            border.color: "#FF5252"
            border.width: 1
            radius: 8
        }

        header: Rectangle {
            width: parent.width
            height: 40
            color: "#FF5252"
            radius: 8
            
            // Mask top corners
            Rectangle {
                width: parent.width
                height: 10
                color: "#FF5252"
                anchors.bottom: parent.bottom
            }

            Text {
                text: errorDialog.title
                anchors.centerIn: parent
                color: "white"
                font.bold: true
            }
        }

        contentItem: ColumnLayout {
            spacing: 20
            Text {
                id: errorText
                Layout.fillWidth: true
                color: "white"
                wrapMode: Text.Wrap
                font.pixelSize: 14
                horizontalAlignment: Text.AlignHCenter
            }
            
            Button {
                text: "OK"
                Layout.alignment: Qt.AlignHCenter
                onClicked: errorDialog.close()
                
                background: Rectangle {
                    implicitWidth: 80
                    implicitHeight: 32
                    color: parent.hovered ? "#333333" : "#222222"
                    radius: 4
                    border.color: "#444444"
                }
                contentItem: Text {
                    text: "OK"
                    color: "white"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    font.bold: true
                }
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 15

        // Header Section
        RowLayout {
            Layout.fillWidth: true
            
            Text {
                text: "Real-time AI Vision"
                color: "#FFFFFF"
                font.pixelSize: 28
                font.bold: true
                Layout.alignment: Qt.AlignLeft
            }
            
            Item { Layout.fillWidth: true } // Spacer
            
            Text {
                text: "Task:"
                color: "#FFFFFF"
                font.pixelSize: 14
            }
            
            ComboBox {
                id: taskComboBox
                model: ["Object Detection", "Pose Estimation", "Image Segmentation"]
                currentIndex: controller.currentTask - 1
                
                onActivated: {
                    if (currentIndex === 0) controller.currentTask = VideoController.TaskObjectDetection
                    else if (currentIndex === 1) controller.currentTask = VideoController.TaskPoseEstimation
                    else if (currentIndex === 2) controller.currentTask = VideoController.TaskImageSegmentation
                }
            }

            Item { Layout.preferredWidth: 10 } // Small spacer

            Text {
                text: "Runtime:"
                color: "#FFFFFF"
                font.pixelSize: 14
            }
            
            ComboBox {
                id: runtimeComboBox
                model: ["OpenVINO", "ONNX Runtime"]
                currentIndex: controller.currentRuntime
                
                onActivated: {
                    if (currentIndex === 0) controller.currentRuntime = VideoController.RuntimeOpenVINO
                    else if (currentIndex === 1) controller.currentRuntime = VideoController.RuntimeONNXRuntime
                }
            }

            Item { Layout.preferredWidth: 10 }

            Text {
                text: "Res:"
                color: "#FFFFFF"
                font.pixelSize: 14
            }

            ComboBox {
                id: resComboBox
                model: controller.supportedResolutions
                currentIndex: {
                    for (var i = 0; i < controller.supportedResolutions.length; i++) {
                        if (controller.supportedResolutions[i].width === controller.currentResolution.width &&
                            controller.supportedResolutions[i].height === controller.currentResolution.height) {
                            return i;
                        }
                    }
                    return -1;
                }
                textRole: "" 
                delegate: ItemDelegate {
                    width: resComboBox.width
                    contentItem: Text {
                        text: modelData.width + "x" + modelData.height
                        color: "white"
                    }
                    background: Rectangle { color: hovered ? "#444444" : "#222222" }
                }
                contentItem: Text {
                    text: controller.currentResolution.width + "x" + controller.currentResolution.height
                    color: "white"
                    verticalAlignment: Text.AlignVCenter
                    leftPadding: 10
                }

                onActivated: {
                    controller.currentResolution = controller.supportedResolutions[currentIndex]
                }
            }
        }

        // Main Content: Video Feed + Metrics Panel
        RowLayout {
            spacing: 20
            Layout.fillWidth: true
            Layout.fillHeight: true

            // Video Container
            Rectangle {
                id: videoContainer
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: "#000000"
                radius: 8
                clip: true

                VideoOutput {
                    id: videoOutput
                    anchors.fill: parent
                    fillMode: VideoOutput.PreserveAspectFit
                }

                // Detection Overlay
                DetectionOverlayItem {
                    id: bboxItem
                    anchors.fill: parent
                    detections: controller.detections

                    Repeater {
                        model: bboxItem.detections
                        Item {
                            id: detectionDelegate
                            property var det: modelData 
                            x: det.x * parent.width
                            y: det.y * parent.height
                            width: det.w * parent.width
                            height: det.h * parent.height
                            
                            Rectangle {
                                id: labelBg
                                y: -height - 2
                                width: labelText.width + 8
                                height: labelText.height + 4
                                color: Qt.hsla((det.classId * 60) % 360 / 360.0, 1.0, 0.5, 0.9)
                                radius: 2
                                transform: Translate {
                                    y: (detectionDelegate.y + labelBg.y < 0) ? labelBg.height + 4 : 0
                                }
                                Text {
                                    id: labelText
                                    anchors.centerIn: parent
                                    text: det.label + " " + Math.round(det.confidence * 100) + "%"
                                    color: "white"
                                    font.pixelSize: 11
                                    font.bold: true
                                }
                            }
                        }
                    }
                }
            }

            // Metrics Panel (Right Sidebar)
            Rectangle {
                id: metricsPanel
                width: 260
                Layout.fillHeight: true
                color: "#1e1e1e"
                border.color: "#333333"
                border.width: 1
                radius: 8

                Column {
                    anchors.fill: parent
                    anchors.margins: 15
                    spacing: 20

                    Text {
                        text: "PERFORMANCE METRICS"
                        color: "#888888"
                        font.pixelSize: 12
                        font.bold: true
                        font.letterSpacing: 1.2
                    }

                    // FPS Section
                    Column {
                        width: parent.width
                        spacing: 8
                        
                        MetricItem {
                            label: "Camera FPS"
                            value: controller.fps.toFixed(1)
                            color: "#00E5FF"
                        }
                        MetricItem {
                            label: "Inference FPS"
                            value: controller.inferenceFps.toFixed(1)
                            color: "#FF00FF"
                        }
                    }

                    Rectangle { width: parent.width; height: 1; color: "#333333" }

                    // Timing Section
                    Column {
                        width: parent.width
                        spacing: 10
                        
                        Text {
                            text: "LATENCY (ms)"
                            color: "#888888"
                            font.pixelSize: 10
                            font.bold: true
                        }
                        
                        MetricItem { label: "Pre-Process"; value: controller.preProcessTime.toFixed(3); color: "#76FF03" }
                        MetricItem { label: "Inference"; value: controller.inferenceTime.toFixed(3); color: "#76FF03" }
                        MetricItem { label: "Post-Process"; value: controller.postProcessTime.toFixed(3); color: "#76FF03" }
                    }

                    Rectangle { width: parent.width; height: 1; color: "#333333" }

                    // System Section
                    Column {
                        width: parent.width
                        spacing: 10
                        
                        Text {
                            text: "SYSTEM RESOURCES"
                            color: "#888888"
                            font.pixelSize: 10
                            font.bold: true
                        }
                        
                        Text {
                            width: parent.width
                            text: controller.systemStats
                            color: "#FFD600"
                            font.family: "Courier"
                            font.pixelSize: 12
                            wrapMode: Text.Wrap
                        }
                    }
                }

                // Helper component for metric rows
                component MetricItem : Row {
                    property string label: ""
                    property string value: ""
                    property color color: "white"
                    width: parent.width
                    
                    Text {
                        text: label
                        color: "#BBBBBB"
                        font.pixelSize: 14
                        width: parent.width * 0.6
                    }
                    Text {
                        text: value
                        color: parent.color
                        font.pixelSize: 16
                        font.bold: true
                        font.family: "Courier New"
                        horizontalAlignment: Text.AlignRight
                        width: parent.width * 0.4
                    }
                }
            }
        }

        // Footer Section
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
