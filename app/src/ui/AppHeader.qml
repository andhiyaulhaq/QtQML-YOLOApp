import QtQuick
import QtQuick.Layouts
import QtQuick.Controls

RowLayout {
    id: root
    Layout.fillWidth: true
    
    property string inputMode: "image"
    property var cameraSource: null
    property var videoSource: null
    property var imageSource: null
    property var detectionController: null
    
    signal sourceChanged(int index)
    signal browseRequested()
    signal taskChanged(int index)
    signal runtimeChanged(int index)
    signal resChanged(int index)

    Text {
        text: "YOLOApp"
        color: "#FFFFFF"
        font.pixelSize: 24
        font.bold: true
    }
    
    Item { Layout.fillWidth: true }

    Text { text: "Source:"; color: "white" }
    CustomComboBox {
        id: sourceCombo
        model: ["Live Camera", "Video File", "Image File"]
        currentIndex: {
            if (inputMode === "camera") return 0;
            if (inputMode === "video") return 1;
            if (inputMode === "image") return 2;
            return 0;
        }
        onActivated: (index) => root.sourceChanged(index)
    }

    // Path Display & Browse Button
    RowLayout {
        visible: inputMode === "video" || inputMode === "image"
        spacing: 8
        
        Rectangle {
            implicitWidth: 200
            implicitHeight: 32
            color: "#1e1e1e"
            border.color: "#333333"
            radius: 4
            clip: true
            
            Text {
                anchors.fill: parent
                anchors.leftMargin: 8
                anchors.rightMargin: 8
                text: {
                    if (inputMode === "video" && videoSource) return videoSource.filePath.split('/').pop();
                    if (inputMode === "image" && imageSource) return imageSource.filePath.split('/').pop();
                    return "";
                }
                color: "#AAAAAA"
                font.pixelSize: 11
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideMiddle
            }
        }

        Button {
            id: browseBtn
            text: "Browse..."
            onClicked: root.browseRequested()
            
            contentItem: Text {
                text: browseBtn.text
                color: "white"
                font.pixelSize: 12
                font.bold: true
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
            
            background: Rectangle {
                implicitWidth: 80
                implicitHeight: 32
                color: browseBtn.hovered ? "#444444" : "#333333"
                border.color: "#555555"
                radius: 4
            }
        }
    }
    
    Text { text: "Task:"; color: "white" }
    CustomComboBox {
        id: taskCombo
        model: ["Detection", "Pose", "Seg"]
        currentIndex: detectionController ? detectionController.currentTask - 1 : 0
        onActivated: (index) => root.taskChanged(index)
    }

    Text { text: "Runtime:"; color: "white" }
    CustomComboBox {
        id: runtimeCombo
        model: ["OpenVINO", "ONNX"]
        currentIndex: detectionController ? detectionController.currentRuntime : 0
        onActivated: (index) => root.runtimeChanged(index)
    }

    Text { 
        text: "Res:"
        color: "white"
        visible: inputMode === "camera"
    }
    CustomComboBox {
        id: resCombo
        visible: inputMode === "camera"
        model: cameraSource ? cameraSource.supportedResolutions : []
        currentIndex: {
            if (!cameraSource) return -1;
            for (var i = 0; i < cameraSource.supportedResolutions.length; i++) {
                if (cameraSource.supportedResolutions[i].width === cameraSource.currentResolution.width &&
                    cameraSource.supportedResolutions[i].height === cameraSource.currentResolution.height) {
                    return i;
                }
            }
            return -1;
        }
        displayText: cameraSource ? cameraSource.currentResolution.width + "x" + cameraSource.currentResolution.height : "0x0"
        onActivated: (index) => root.resChanged(index)
    }
}
