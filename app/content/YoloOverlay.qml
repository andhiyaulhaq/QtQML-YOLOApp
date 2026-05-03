import QtQuick
import CameraModule 1.0

DetectionOverlayItem {
    id: overlay
    
    property var detectionController: null // Link to detection controller
    detections: detectionController ? detectionController.detections : null
    
    property real videoAspectRatio: detectionController && detectionController.detections && detectionController.detections.frameSize.height > 0 ? 
                                    detectionController.detections.frameSize.width / detectionController.detections.frameSize.height : 1.0
    property real itemAspectRatio: height > 0 ? width / height : 1.0
    property real renderW: videoAspectRatio > itemAspectRatio ? width : height * videoAspectRatio
    property real renderH: videoAspectRatio > itemAspectRatio ? width / videoAspectRatio : height
    property real offsetX: (width - renderW) / 2.0
    property real offsetY: (height - renderH) / 2.0
    
    Repeater {
        model: overlay.detections
        Item {
            x: overlay.offsetX + modelData.x * overlay.renderW
            y: overlay.offsetY + modelData.y * overlay.renderH
            width: modelData.w * overlay.renderW
            height: modelData.h * overlay.renderH
            
            Rectangle {
                y: -20
                width: labelTxt.width + 10
                height: 18
                color: Qt.hsla((modelData.classId * 60) % 360 / 360.0, 1.0, 0.5, 0.8)
                radius: 2
                Text {
                    id: labelTxt
                    anchors.centerIn: parent
                    text: modelData.label + " " + Math.round(modelData.confidence * 100) + "%"
                    color: "white"
                    font.pixelSize: 10
                    font.bold: true
                }
            }
        }
    }
}
