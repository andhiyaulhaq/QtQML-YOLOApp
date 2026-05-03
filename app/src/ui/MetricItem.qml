import QtQuick

Row {
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
