import QtQuick
import QtQuick.Controls
import CameraModule 1.0

Window {
    width: 800
    height: 600
    visible: true
    title: qsTr("OpenCV Qt6 Camera")
    color: "#2b2b2b"

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

        OpenCVItem {
            id: camera
            width: 640
            height: 480
            anchors.horizontalCenter: parent.horizontalCenter

            Rectangle {
                anchors.fill: parent
                color: "transparent"
                border.color: "#41cd52"
                border.width: 2
            }

            // --- FPS Display Overlay ---
            Rectangle {
                width: 80
                height: 30
                color: "#aa000000" // Semi-transparent black
                radius: 5
                anchors.top: parent.top
                anchors.left: parent.left
                anchors.margins: 10

                Text {
                    anchors.centerIn: parent
                    // Bind directly to the C++ property
                    text: "FPS: " + camera.fps
                    color: "#00ff00"
                    font.bold: true
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
