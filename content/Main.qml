import QtQuick
import QtQuick.Controls
import CameraModule 1.0 // Import our C++ module

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

        // This is our C++ Class!
        OpenCVItem {
            id: camera
            width: 640
            height: 480
            // Center it horizontally
            anchors.horizontalCenter: parent.horizontalCenter

            // Add a simple border
            Rectangle {
                anchors.fill: parent
                color: "transparent"
                border.color: "#41cd52" // Green border
                border.width: 2
            }
        }

        Button {
            text: "Close App"
            anchors.horizontalCenter: parent.horizontalCenter
            onClicked: Qt.quit()
        }
    }
}
