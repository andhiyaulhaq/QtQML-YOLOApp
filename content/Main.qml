import QtQuick
import QtQuick.Controls
import QtMultimedia 6.0
import CameraModule 1.0

Window {
    width: 800
    height: 600
    visible: true
    title: "Qt6 Optimized Camera (Async Inference)"
    color: "#2b2b2b"

    VideoController {
        id: controller
        videoSink: videoOutput.videoSink
    }

    Column {
        anchors.fill: parent
        spacing: 10
        padding: 20

        Text {
            text: "Live Feed"
            color: "white"
            font.pixelSize: 24
            font.bold: true
        }

        Item {
            width: 640
            height: 480
            anchors.horizontalCenter: parent.horizontalCenter

            VideoOutput {
                id: videoOutput
                anchors.fill: parent
                fillMode: VideoOutput.PreserveAspectFit
            }

            // Optimized Bounding Box Overlay (C++ Scene Graph for Boxes)
            BoundingBoxItem {
                id: bboxItem
                anchors.fill: parent
                detections: controller.detections

                // Hybrid Approach: Use QML Repeater for Text Labels (High Performance & Easy Text Handling)
                Repeater {
                    model: bboxItem.detections
                    
                    Item {
                        id: detectionDelegate
                        // Bind to detection data
                        // modelData is the Detection struct (Q_GADGET)
                        property var det: modelData 
                        
                        x: det.x * parent.width
                        y: det.y * parent.height
                        width: det.w * parent.width
                        height: det.h * parent.height
                        
                        // Label Background
                        Rectangle {
                            id: labelBg
                            y: -height - 2
                            width: labelText.width + 8
                            height: labelText.height + 4
                            color: {
                                // Calculate color based on classId (same logic as C++)
                                var hue = (det.classId * 60) % 360
                                return Qt.hsla(hue / 360.0, 1.0, 0.5, 1.0)
                            }
                            radius: 2
                            
                            // Flip label if at top edge
                            transform: Translate {
                                // Check if label goes above top of the container (detectionDelegate.y is relative to BoundingBoxItem)
                                y: (detectionDelegate.y + labelBg.y < 0) ? labelBg.height + 4 : 0
                            }

                            Text {
                                id: labelText
                                anchors.centerIn: parent
                                text: det.label + " " + Math.round(det.confidence * 100) + "%"
                                color: "black"
                                font.pixelSize: 12
                                font.bold: true
                            }
                        }
                    }
                }
            }
        
            // Performance Overlay
            Column {
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.margins: 10
                spacing: 5
                
                Text {
                    text: "Camera FPS: " + controller.fps.toFixed(1)
                    color: "cyan"
                    font.pixelSize: 18
                    font.bold: true
                    style: Text.Outline
                    styleColor: "black"
                }

                Text {
                    text: "Inf FPS: " + controller.inferenceFps.toFixed(1)
                    color: "magenta"
                    font.pixelSize: 18
                    font.bold: true
                    style: Text.Outline
                    styleColor: "black"
                }
                
                Text {
                    text: controller.systemStats
                    color: "yellow"
                    font.pixelSize: 14
                    font.bold: true
                    style: Text.Outline
                    styleColor: "black"
                }

                Text {
                    text: "Pre: " + controller.preProcessTime.toFixed(1) + " ms"
                    color: "lightgreen"
                    font.pixelSize: 14
                    font.bold: true
                    style: Text.Outline
                    styleColor: "black"
                }

                Text {
                    text: "Infer: " + controller.inferenceTime.toFixed(1) + " ms"
                    color: "lightgreen"
                    font.pixelSize: 14
                    font.bold: true
                    style: Text.Outline
                    styleColor: "black"
                }

                Text {
                    text: "Post: " + controller.postProcessTime.toFixed(1) + " ms"
                    color: "lightgreen"
                    font.pixelSize: 14
                    font.bold: true
                    style: Text.Outline
                    styleColor: "black"
                }
            }
        }
    
        Button {
            text: "Close App"
            anchors.horizontalCenter: parent.horizontalCenter
            onClicked: Qt.quit()
        }
    }
}
