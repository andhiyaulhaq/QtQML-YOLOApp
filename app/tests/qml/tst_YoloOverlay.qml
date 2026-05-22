import QtQuick 2.0
import QtQuick.Layouts 1.3
import QtQuick.Controls 2.15
import QtTest
import CameraModule 1.0 
import "../../src/ui"

TestCase {
    id: testCase
    name: "YoloOverlayTests"
    when: windowShown

    Item { id: testRoot }

    // Mock InferenceController for testing purposes
    QtObject {
        id: mockInferenceController
        property var visionObjects: QtObject {
            property var frameSize: Qt.size(0, 0)
        }
    }

    Component {
        id: yoloOverlayComponent
        YoloOverlay {
            id: overlay
            width: 800
            height: 600
            inferenceController: mockInferenceController
        }
    }

    // Helper function to create an overlay instance for each test case
    function createOverlay() {
        var overlay = yoloOverlayComponent.createObject(testRoot);
        return overlay;
    }

    function cleanup() {
        // Destroy any created objects after each test
        for (var i = testRoot.children.length - 1; i >= 0; --i) {
            testRoot.children[i].destroy();
        }
    }

    function test_aspectRatioCalculation_wideVideo_narrowItem() {
        var overlay = createOverlay();
        overlay.width = 800;
        overlay.height = 600; // Item aspect ratio = 800/600 = 1.33
        mockInferenceController.visionObjects.frameSize = Qt.size(1920, 1080); // Video aspect ratio = 1920/1080 = 1.77

        // videoAspectRatio (1.77) > itemAspectRatio (1.33)
        // renderH = itemW / videoAspectRatio = 800 / 1.777 = 450
        // renderW = itemW = 800
        // offsetY = (itemH - renderH) / 2 = (600 - 450) / 2 = 75
        // offsetX = 0
        
        compare(overlay.renderW, 800);
        verify(Math.abs(overlay.renderH - 450) < 1); // Allow small floating point deviation
        compare(overlay.offsetX, 0);
        verify(Math.abs(overlay.offsetY - 75) < 1);
    }

    function test_aspectRatioCalculation_narrowVideo_wideItem() {
        var overlay = createOverlay();
        overlay.width = 1600;
        overlay.height = 900; // Item aspect ratio = 1600/900 = 1.77
        mockInferenceController.visionObjects.frameSize = Qt.size(800, 600); // Video aspect ratio = 800/600 = 1.33

        // videoAspectRatio (1.33) < itemAspectRatio (1.77)
        // renderW = itemH * videoAspectRatio = 900 * 1.333 = 1200
        // renderH = itemH = 900
        // offsetX = (itemW - renderW) / 2 = (1600 - 1200) / 2 = 200
        // offsetY = 0

        verify(Math.abs(overlay.renderW - 1200) < 1);
        compare(overlay.renderH, 900);
        verify(Math.abs(overlay.offsetX - 200) < 1);
        compare(overlay.offsetY, 0);
    }
    
    function test_aspectRatioCalculation_square() {
        var overlay = createOverlay();
        overlay.width = 500;
        overlay.height = 500; // Item aspect ratio = 1
        mockInferenceController.visionObjects.frameSize = Qt.size(1000, 1000); // Video aspect ratio = 1

        compare(overlay.renderW, 500);
        compare(overlay.renderH, 500);
        compare(overlay.offsetX, 0);
        compare(overlay.offsetY, 0);
    }
    
    function test_noFrameSize() {
        var overlay = createOverlay();
        overlay.width = 800;
        overlay.height = 600;
        mockInferenceController.visionObjects.frameSize = Qt.size(0, 0); // No frame size yet

        // Implementation defaults to videoAspectRatio = 1.0 if height is 0 or empty
        // videoAspectRatio (1.0) < itemAspectRatio (1.33) -> renderW = height * 1.0 = 600
        compare(overlay.renderW, 600);
        compare(overlay.renderH, 600);
        compare(overlay.offsetX, 100); // (800 - 600) / 2
        compare(overlay.offsetY, 0);
    }
}
