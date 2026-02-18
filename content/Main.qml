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

            // Bounding Box Overlay
            Repeater {
                model: controller.detections
                delegate: Item {
                    // Normalize coordinates are 0.0-1.0 relative to 640x480 source
                    // We need to map them to the VideoOutput's actual displayed rect
                    // For Stretch/PreserveAspectFit, it might differ. 
                    // Since container matches source aspect (640x480), it's simple mapping.
                    
                    x: modelData.x * parent.width
                    y: modelData.y * parent.height
                    width: modelData.w * parent.width
                    height: modelData.h * parent.height

                    Rectangle {
                        anchors.fill: parent
                        color: "transparent"
                        border.color: "lime"
                        border.width: 2
                    }
                    
                    Rectangle {
                        color: "lime"
                        x: 0
                        y: -20
                        width: labelText.contentWidth + 10
                        height: 20
                        visible: y >= 0
                        
                        Text {
                            id: labelText
                            anchors.centerIn: parent
                            text: modelData.label + " " + (modelData.confidence * 100).toFixed(0) + "%"
                            color: "black"
                            font.pixelSize: 12
                            font.bold: true
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
                    text: "FPS: " + controller.fps.toFixed(1)
                    color: "cyan"
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
            }
        }
    
        Button {
            text: "Close App"
            anchors.horizontalCenter: parent.horizontalCenter
            onClicked: Qt.quit()
        }
    }
}
