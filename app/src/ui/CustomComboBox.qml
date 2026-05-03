import QtQuick
import QtQuick.Controls

ComboBox {
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
            radius: 4
            anchors.fill: parent
            anchors.margins: 2
            visible: delegateItem.highlighted
        }
    }
}
