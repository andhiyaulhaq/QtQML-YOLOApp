#pragma once

#include <QSize>

struct RenderTransform {
    float renderW, renderH;
    float offsetX, offsetY;

    static inline RenderTransform calculate(float itemW, float itemH, const QSize& frameSize) {
        RenderTransform t;
        t.renderW = itemW;
        t.renderH = itemH;
        t.offsetX = 0;
        t.offsetY = 0;

        if (!frameSize.isEmpty() && itemW > 0 && itemH > 0) {
            float videoAspectRatio = static_cast<float>(frameSize.width()) / static_cast<float>(frameSize.height());
            float itemAspectRatio = itemW / itemH;

            if (videoAspectRatio > itemAspectRatio) {
                t.renderH = itemW / videoAspectRatio;
                t.offsetY = (itemH - t.renderH) / 2.0f;
            } else {
                t.renderW = itemH * videoAspectRatio;
                t.offsetX = (itemW - t.renderW) / 2.0f;
            }
        }
        return t;
    }
};
