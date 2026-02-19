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

            // Optimized Bounding Box Overlay
            BoundingBoxItem {
                anchors.fill: parent
                detections: controller.detections
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
