import QtQuick
import QtQuick.Layouts

Rectangle {
    id: root
    width: 260
    color: "#1e1e1e"
    border.color: "#333333"
    border.width: 1
    radius: 8

    property string inputMode: ""
    property var cameraSource: null
    property var inferenceController: null
    property var monitoringSource: null

    Column {
        anchors.fill: parent
        anchors.margins: 15
        spacing: 20

        Text {
            text: "PERFORMANCE METRICS"
            color: "#888888"
            font.pixelSize: 12
            font.bold: true
            font.letterSpacing: 1.2
        }

        Column {
            width: parent.width
            spacing: 8
            
            MetricItem {
                label: "Input FPS"
                value: (inputMode === "image" || !cameraSource) ? "-" : cameraSource.cameraFps.toFixed(1)
                color: "#00E5FF"
            }
            MetricItem {
                label: "Inference FPS"
                value: (inputMode === "image" || !inferenceController) ? "-" : inferenceController.inferenceFps.toFixed(1)
                color: "#FF00FF"
            }
        }

        Rectangle { width: parent.width; height: 1; color: "#333333" }

        Column {
            width: parent.width
            spacing: 10
            
            Text {
                text: "LATENCY (ms)"
                color: "#888888"
                font.pixelSize: 10
                font.bold: true
            }
            
            MetricItem { label: "Pre-Process"; value: inferenceController ? inferenceController.preProcessTime.toFixed(3) : "0.000"; color: "#76FF03" }
            MetricItem { label: "Inference"; value: inferenceController ? inferenceController.inferenceTime.toFixed(3) : "0.000"; color: "#76FF03" }
            MetricItem { label: "Post-Process"; value: inferenceController ? inferenceController.postProcessTime.toFixed(3) : "0.000"; color: "#76FF03" }
        }

        Rectangle { width: parent.width; height: 1; color: "#333333" }

        Column {
            width: parent.width
            spacing: 10
            
            Text {
                text: "SYSTEM RESOURCES"
                color: "#888888"
                font.pixelSize: 10
                font.bold: true
            }
            
            Text {
                width: parent.width
                text: monitoringSource ? monitoringSource.statsText : "Monitoring not available"
                color: "#FFD600"
                font.family: "Courier"
                font.pixelSize: 12
                wrapMode: Text.Wrap
            }
        }
    }
}
