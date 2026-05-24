import QtQuick
import QtQuick.Layouts
import QtQuick.Controls

Rectangle {
    id: root
    height: 40
    color: "#AA000000"
    
    property var videoSource: null // Link to video controller
    
    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 15
        anchors.rightMargin: 15
        spacing: 10
        
        Text {
            text: (typeof videoSource !== "undefined" && videoSource) ? videoSource.currentTimeStr : "00:00"
            color: "white"
            font.pixelSize: 12
            font.family: "Courier"
        }
        
        Slider {
            id: seekSlider
            Layout.fillWidth: true
            from: 0
            to: 1.0
            value: (typeof videoSource !== "undefined" && videoSource && videoSource.totalFrames > 0) ? videoSource.currentFrame / videoSource.totalFrames : 0
            hoverEnabled: true

            MouseArea {
                anchors.fill: parent
                hoverEnabled: true
                acceptedButtons: Qt.NoButton
                cursorShape: Qt.PointingHandCursor
            }
            
            onMoved: {
                if (typeof videoSource !== "undefined" && videoSource) videoSource.seek(value)
            }
            
            background: Rectangle {
                x: seekSlider.leftPadding
                y: seekSlider.topPadding + seekSlider.availableHeight / 2 - height / 2
                implicitWidth: 200
                implicitHeight: seekSlider.hovered ? 6 : 4
                width: seekSlider.availableWidth
                height: implicitHeight
                radius: height / 2
                color: "#333333"

                Behavior on implicitHeight {
                    NumberAnimation { duration: 150 }
                }

                Rectangle {
                    width: seekSlider.visualPosition * parent.width
                    height: parent.height
                    color: "#00E5FF"
                    radius: height / 2
                }
            }
            
            handle: Rectangle {
                x: seekSlider.leftPadding + seekSlider.visualPosition * (seekSlider.availableWidth - width)
                y: seekSlider.topPadding + seekSlider.availableHeight / 2 - height / 2
                implicitWidth: seekSlider.hovered || seekSlider.pressed ? 14 : 8
                implicitHeight: seekSlider.hovered || seekSlider.pressed ? 14 : 8
                radius: width / 2
                color: seekSlider.pressed ? "#00B8D4" : "#00E5FF"
                border.color: "white"
                border.width: 1

                Behavior on implicitWidth {
                    NumberAnimation { duration: 150 }
                }
                Behavior on implicitHeight {
                    NumberAnimation { duration: 150 }
                }
            }
        }
        
        Text {
            text: (typeof videoSource !== "undefined" && videoSource) ? videoSource.totalTimeStr : "00:00"
            color: "white"
            font.pixelSize: 12
            font.family: "Courier"
        }
    }
}
