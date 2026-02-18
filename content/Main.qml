import QtQuick
import QtQuick.Controls
import QtMultimedia 6.0
import CameraModule 1.0

Window {
    width: 800
    height: 600
    visible: true
    title: "Qt6 Optimized Camera"
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

        VideoOutput {
            id: videoOutput
            width: 640
            height: 480
            anchors.horizontalCenter: parent.horizontalCenter
            fillMode: VideoOutput.PreserveAspectFit

        // Performance Overlay
        Rectangle {
            anchors.fill: parent
            color: "transparent"
            border.color: "transparent" // Make border transparent for now
            // border.width: 2
            
            Column {
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.margins: 10
                spacing: 5
                
                Text {
                    text: "FPS: " + controller.fps.toFixed(1)
                    color: "lime" // High visibility color
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
}
