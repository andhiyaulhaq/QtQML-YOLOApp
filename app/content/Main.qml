import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import QtMultimedia 6.0
import CameraModule 1.0

Window {
    width: 1000
    height: 700
    visible: true
    title: "YOLOApp - Dual Input Source"
    color: "#121212"

    property string inputMode: "image" // "camera", "video", "image"

    FileDialog {
        id: videoFileDialog
        title: "Select Video File"
        nameFilters: ["Video files (*.mp4 *.avi *.mkv *.mov *.wmv)"]
        onAccepted: {
            inputMode = "video"
            videoFile.setFilePath(selectedFile)
        }
        onRejected: {
            if (inputMode === "video" && !videoFile.hasFile) {
                sourceCombo.currentIndex = 0
                inputMode = "camera"
                camera.activate()
            }
        }
    }

    FileDialog {
        id: imageFileDialog
        title: "Select Image File"
        nameFilters: ["Image files (*.jpg *.jpeg *.png *.bmp)"]
        onAccepted: {
            inputMode = "image"
            imageFile.setFilePath(selectedFile)
        }
        onRejected: {
            if (inputMode === "image" && !imageFile.hasFile) {
                sourceCombo.currentIndex = 0
                inputMode = "camera"
                camera.activate()
            }
        }
    }

    component StyledComboBox : ComboBox {
        id: control
        background: Rectangle {
            implicitWidth: 140
            implicitHeight: 32
            color: "#1e1e1e"
            border.color: control.hovered || control.pressed ? "#555555" : "#333333"
            border.width: 1
            radius: 4
        }
        contentItem: Text {
            leftPadding: 10
            rightPadding: control.indicator.width + control.spacing
            text: control.displayText
            font: control.font
            color: "white"
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }
        indicator: Canvas {
            id: canvas
            x: control.width - width - 10
            y: (control.height - height) / 2
            width: 10
            height: 6
            contextType: "2d"
            onPaint: {
                var context = getContext("2d");
                context.reset();
                context.lineWidth = 1.5;
                context.strokeStyle = "white";
                context.beginPath();
                context.moveTo(1, 1);
                context.lineTo(width / 2, height - 1);
                context.lineTo(width - 1, 1);
                context.stroke();
            }
            Connections {
                target: control
                function onPressedChanged() { canvas.requestPaint(); }
            }
        }
        popup: Popup {
            y: control.height + 2
            width: control.width
            implicitHeight: Math.min(contentItem.implicitHeight, 200)
            padding: 0
            contentItem: ListView {
                clip: true
                implicitHeight: contentHeight
                model: control.popup.visible ? control.delegateModel : null
                currentIndex: control.highlightedIndex
                spacing: 0
                ScrollIndicator.vertical: ScrollIndicator { }
            }
            background: Rectangle {
                color: "#1e1e1e"
                border.color: "#333333"
                radius: 4
            }
        }
        delegate: ItemDelegate {
            id: delegateItem
            width: control.width
            implicitHeight: index === control.currentIndex ? 0 : 32
            visible: index !== control.currentIndex
            padding: 0
            highlighted: control.highlightedIndex === index
            contentItem: Text {
                visible: delegateItem.visible
                text: control.textRole ? (Array.isArray(control.model) ? modelData[control.textRole] : model[control.textRole]) : (modelData.width ? modelData.width + "x" + modelData.height : modelData)
                color: "white"
                font: control.font
                elide: Text.ElideRight
                verticalAlignment: Text.AlignVCenter
                leftPadding: 10
            }
            background: Rectangle {
                color: delegateItem.highlighted ? "#333333" : "transparent"
                visible: delegateItem.highlighted
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 15

        // Header Section
        RowLayout {
            Layout.fillWidth: true
            Text {
                text: "YOLO Dual Source"
                color: "#FFFFFF"
                font.pixelSize: 24
                font.bold: true
            }
            Item { Layout.fillWidth: true }

            Text { text: "Source:"; color: "white" }
            StyledComboBox {
                id: sourceCombo
                model: ["Live Camera", "Video File", "Image File"]
                currentIndex: {
                    if (inputMode === "camera") return 0;
                    if (inputMode === "video") return 1;
                    if (inputMode === "image") return 2;
                    return 0;
                }
                onActivated: (index) => {
                    if (index === 0) {
                        inputMode = "camera"
                        camera.activate()
                    } else if (index === 1) {
                        if (videoFile.hasFile) {
                            inputMode = "video"
                            videoFile.activate()
                        } else {
                            videoFileDialog.open()
                        }
                    } else if (index === 2) {
                        if (imageFile.hasFile) {
                            inputMode = "image"
                            imageFile.activate()
                        } else {
                            imageFileDialog.open()
                        }
                    }
                }
            }

            // Path Display & Browse Button (for Video/Image mode)
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
                            if (inputMode === "video" && videoFile) return videoFile.filePath.split('/').pop();
                            if (inputMode === "image" && imageFile) return imageFile.filePath.split('/').pop();
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
                    onClicked: {
                        if (inputMode === "video") videoFileDialog.open();
                        else if (inputMode === "image") imageFileDialog.open();
                    }
                    
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
            StyledComboBox {
                id: taskCombo
                model: ["Detection", "Pose", "Seg"]
                currentIndex: detection ? detection.currentTask - 1 : 0
                onActivated: (index) => {
                    if (!detection) return;
                    if (index === 0) detection.currentTask = YoloTask.ObjectDetection
                    else if (index === 1) detection.currentTask = YoloTask.PoseEstimation
                    else if (index === 2) detection.currentTask = YoloTask.ImageSegmentation
                }
            }

            Text { text: "Runtime:"; color: "white" }
            StyledComboBox {
                id: runtimeCombo
                model: ["OpenVINO", "ONNX"]
                currentIndex: detection ? detection.currentRuntime : 0
                onActivated: (index) => {
                    if (!detection) return;
                    if (index === 0) detection.currentRuntime = YoloTask.OpenVINO
                    else if (index === 1) detection.currentRuntime = YoloTask.ONNXRuntime
                }
            }

            Text { 
                text: "Res:"
                color: "white"
                visible: inputMode === "camera"
            }
            StyledComboBox {
                id: resCombo
                visible: inputMode === "camera"
                model: camera ? camera.supportedResolutions : []
                currentIndex: {
                    if (!camera) return -1;
                    for (var i = 0; i < camera.supportedResolutions.length; i++) {
                        if (camera.supportedResolutions[i].width === camera.currentResolution.width &&
                            camera.supportedResolutions[i].height === camera.currentResolution.height) {
                            return i;
                        }
                    }
                    return -1;
                }
                displayText: camera ? camera.currentResolution.width + "x" + camera.currentResolution.height : "0x0"
                onActivated: (index) => {
                    if (camera) {
                        var res = camera.supportedResolutions[index]
                        camera.currentResolution = res
                    }
                }
            }
        }

        // Main Layout
        RowLayout {
            spacing: 20
            Layout.fillWidth: true
            Layout.fillHeight: true

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: "black"
                radius: 8
                clip: true

                VideoOutput {
                    id: videoOutput
                    anchors.fill: parent
                    fillMode: VideoOutput.PreserveAspectFit
                    Component.onCompleted: {
                        if (camera) camera.videoSink = videoOutput.videoSink
                    }
                }

                DetectionOverlayItem {
                    id: overlay
                    anchors.fill: videoOutput
                    detections: detection ? detection.detections : null
                    
                    property real videoAspectRatio: detection && detection.detections && detection.detections.frameSize.height > 0 ? 
                                                    detection.detections.frameSize.width / detection.detections.frameSize.height : 1.0
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

                // Seek Bar for Video File
                Rectangle {
                    anchors.bottom: parent.bottom
                    anchors.left: parent.left
                    anchors.right: parent.right
                    height: 40
                    color: "#AA000000"
                    visible: inputMode === "video"
                    
                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 15
                        anchors.rightMargin: 15
                        spacing: 10
                        
                        Text {
                            text: (typeof videoFile !== "undefined" && videoFile) ? videoFile.currentTimeStr : "00:00"
                            color: "white"
                            font.pixelSize: 12
                            font.family: "Courier"
                        }
                        
                        Slider {
                            id: seekSlider
                            Layout.fillWidth: true
                            from: 0
                            to: 1.0
                            value: (typeof videoFile !== "undefined" && videoFile && videoFile.totalFrames > 0) ? videoFile.currentFrame / videoFile.totalFrames : 0
                            
                            onMoved: {
                                if (typeof videoFile !== "undefined" && videoFile) videoFile.seek(value)
                            }
                            
                            background: Rectangle {
                                x: seekSlider.leftPadding
                                y: seekSlider.topPadding + seekSlider.availableHeight / 2 - height / 2
                                implicitWidth: 200
                                implicitHeight: 4
                                width: seekSlider.availableWidth
                                height: implicitHeight
                                radius: 2
                                color: "#333333"

                                Rectangle {
                                    width: seekSlider.visualPosition * parent.width
                                    height: parent.height
                                    color: "#00E5FF"
                                    radius: 2
                                }
                            }
                            
                            handle: Rectangle {
                                x: seekSlider.leftPadding + seekSlider.visualPosition * (seekSlider.availableWidth - width)
                                y: seekSlider.topPadding + seekSlider.availableHeight / 2 - height / 2
                                implicitWidth: 12
                                implicitHeight: 12
                                radius: 6
                                color: seekSlider.pressed ? "#00B8D4" : "#00E5FF"
                                border.color: "white"
                                border.width: 1
                            }
                        }
                        
                        Text {
                            text: (typeof videoFile !== "undefined" && videoFile) ? videoFile.totalTimeStr : "00:00"
                            color: "white"
                            font.pixelSize: 12
                            font.family: "Courier"
                        }
                    }
                }
            }

            // Metrics Panel
            Rectangle {
                id: metricsPanel
                width: 260
                Layout.fillHeight: true
                color: "#1e1e1e"
                border.color: "#333333"
                border.width: 1
                radius: 8

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
                            value: camera ? camera.cameraFps.toFixed(1) : "0.0"
                            color: "#00E5FF"
                        }
                        MetricItem {
                            label: "Inference FPS"
                            value: detection ? detection.inferenceFps.toFixed(1) : "0.0"
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
                        
                        MetricItem { label: "Pre-Process"; value: detection ? detection.preProcessTime.toFixed(3) : "0.000"; color: "#76FF03" }
                        MetricItem { label: "Inference"; value: detection ? detection.inferenceTime.toFixed(3) : "0.000"; color: "#76FF03" }
                        MetricItem { label: "Post-Process"; value: detection ? detection.postProcessTime.toFixed(3) : "0.000"; color: "#76FF03" }
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
                            text: monitoring ? monitoring.statsText : "Monitoring not available"
                            color: "#FFD600"
                            font.family: "Courier"
                            font.pixelSize: 12
                            wrapMode: Text.Wrap
                        }
                    }
                }

                component MetricItem : Row {
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
            }
        }

        // Footer
        Button {
            text: "Exit Application"
            Layout.alignment: Qt.AlignHCenter
            onClicked: Qt.quit()
            
            contentItem: Text {
                text: parent.text
                color: parent.pressed ? "#FF5252" : "#FFFFFF"
                font.bold: true
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
            
            background: Rectangle {
                implicitWidth: 150
                implicitHeight: 40
                color: parent.hovered ? "#333333" : "#222222"
                radius: 4
                border.color: "#444444"
                border.width: 1
            }
        }
    }
}
