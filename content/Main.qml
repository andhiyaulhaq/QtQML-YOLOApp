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
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 15

        // Header Section
        Text {
            text: "Real-time AI Vision"
            color: "#FFFFFF"
            font.pixelSize: 28
            font.bold: true
            Layout.alignment: Qt.AlignLeft
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

                // Bounding Box Overlay
                BoundingBoxItem {
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
