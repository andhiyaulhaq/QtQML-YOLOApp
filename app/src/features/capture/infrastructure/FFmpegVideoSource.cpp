#include "FFmpegVideoSource.h"
#include <QDebug>

FFmpegVideoSource::FFmpegVideoSource() {
    m_packet = av_packet_alloc();
    m_frame = av_frame_alloc();
    m_frameRGB = av_frame_alloc();
}

FFmpegVideoSource::~FFmpegVideoSource() {
    cleanup();
    av_packet_free(&m_packet);
    av_frame_free(&m_frame);
    av_frame_free(&m_frameRGB);
}

bool FFmpegVideoSource::open(const SourceConfig& config) {
    cleanup();
    return initFFmpeg(config.filePath.toStdString());
}

void FFmpegVideoSource::close() {
    cleanup();
}

bool FFmpegVideoSource::initFFmpeg(const std::string& path) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (avformat_open_input(&m_formatCtx, path.c_str(), nullptr, nullptr) != 0) {
        qDebug() << "[FFmpeg] Could not open file:" << path.c_str();
        return false;
    }

    if (avformat_find_stream_info(m_formatCtx, nullptr) < 0) {
        qDebug() << "[FFmpeg] Could not find stream info";
        return false;
    }

    m_videoStreamIndex = -1;
    for (unsigned int i = 0; i < m_formatCtx->nb_streams; i++) {
        if (m_formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            m_videoStreamIndex = i;
            break;
        }
    }

    if (m_videoStreamIndex == -1) return false;

    const AVCodec* codec = avcodec_find_decoder(m_formatCtx->streams[m_videoStreamIndex]->codecpar->codec_id);
    if (!codec) return false;

    m_codecCtx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(m_codecCtx, m_formatCtx->streams[m_videoStreamIndex]->codecpar);

    // Set threading for performance
    m_codecCtx->thread_count = 0; // Auto detect

    if (avcodec_open2(m_codecCtx, codec, nullptr) < 0) return false;

    AVStream* stream = m_formatCtx->streams[m_videoStreamIndex];
    m_resolution = QSize(m_codecCtx->width, m_codecCtx->height);
    m_fps = av_q2d(stream->avg_frame_rate);
    
    // Use duration for more accurate total frame count
    if (m_formatCtx->duration != AV_NOPTS_VALUE) {
        m_totalFrames = av_rescale_q(m_formatCtx->duration, AVRational{1, AV_TIME_BASE}, av_inv_q(stream->avg_frame_rate));
    } else {
        m_totalFrames = stream->nb_frames;
    }

    // Detect Rotation using side data (modern FFmpeg 7/8 way)
    const AVPacketSideData *sd = av_packet_side_data_get(stream->codecpar->coded_side_data, stream->codecpar->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX);
    if (sd) {
        m_rotation = (int)-av_display_rotation_get((const int32_t*)sd->data);
    } else {
        AVDictionaryEntry *tag = av_dict_get(stream->metadata, "rotate", nullptr, 0);
        if (tag) m_rotation = atoi(tag->value);
        else m_rotation = 0;
    }
    m_rotation = (m_rotation + 360) % 360;

    if (m_rotation == 90 || m_rotation == 270) {
        m_resolution = QSize(m_codecCtx->height, m_codecCtx->width);
    }

    qDebug() << "[FFmpeg] Opened:" << path.c_str() << "Res:" << m_resolution << "FPS:" << m_fps << "Frames:" << m_totalFrames << "Rot:" << m_rotation;

    // Allocate RGB buffer
    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, m_codecCtx->width, m_codecCtx->height, 1);
    m_buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));

    av_image_fill_arrays(m_frameRGB->data, m_frameRGB->linesize, m_buffer, AV_PIX_FMT_BGR24, 
                         m_codecCtx->width, m_codecCtx->height, 1);

    m_swsCtx = sws_getContext(m_codecCtx->width, m_codecCtx->height, m_codecCtx->pix_fmt,
                              m_codecCtx->width, m_codecCtx->height, AV_PIX_FMT_BGR24,
                              SWS_BICUBIC, nullptr, nullptr, nullptr);

    qDebug() << "[FFmpeg] Opened:" << path.c_str() << "Res:" << m_resolution << "FPS:" << m_fps;
    return true;
}

void FFmpegVideoSource::cleanup() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_swsCtx) { sws_freeContext(m_swsCtx); m_swsCtx = nullptr; }
    if (m_codecCtx) { avcodec_free_context(&m_codecCtx); m_codecCtx = nullptr; }
    if (m_formatCtx) { avformat_close_input(&m_formatCtx); m_formatCtx = nullptr; }
    if (m_buffer) { av_free(m_buffer); m_buffer = nullptr; }
    m_currentFrameIndex = 0;
}

bool FFmpegVideoSource::readFrame(cv::Mat& outFrame) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return readFrameInternal(outFrame);
}

bool FFmpegVideoSource::readFrameInternal(cv::Mat& outFrame) {
    if (!m_codecCtx) return false;

    bool frameFound = false;
    while (!frameFound) {
        int ret = av_read_frame(m_formatCtx, m_packet);
        if (ret >= 0) {
            if (m_packet->stream_index == m_videoStreamIndex) {
                if (avcodec_send_packet(m_codecCtx, m_packet) >= 0) {
                    if (avcodec_receive_frame(m_codecCtx, m_frame) >= 0) {
                        frameFound = true;
                    }
                }
            }
            av_packet_unref(m_packet);
        } else {
            // EOF reached - Flush the decoder to get last buffered frames
            avcodec_send_packet(m_codecCtx, nullptr);
            if (avcodec_receive_frame(m_codecCtx, m_frame) >= 0) {
                frameFound = true;
            } else {
                return false; // Actually EOF
            }
        }
    }

    if (frameFound) {
        // Convert to BGR for OpenCV compatibility
        sws_scale(m_swsCtx, (uint8_t const* const*)m_frame->data, m_frame->linesize, 0, m_codecCtx->height,
                    m_frameRGB->data, m_frameRGB->linesize);

        // Copy to cv::Mat
        outFrame = cv::Mat(m_codecCtx->height, m_codecCtx->width, CV_8UC3, m_frameRGB->data[0]).clone();

        // Apply Rotation if needed
        if (m_rotation == 90) {
            cv::rotate(outFrame, outFrame, cv::ROTATE_90_CLOCKWISE);
        } else if (m_rotation == 180) {
            cv::rotate(outFrame, outFrame, cv::ROTATE_180);
        } else if (m_rotation == 270) {
            cv::rotate(outFrame, outFrame, cv::ROTATE_90_COUNTERCLOCKWISE);
        }

        // Sync frame index using PTS for accuracy
        AVStream* stream = m_formatCtx->streams[m_videoStreamIndex];
        if (m_frame->pts != AV_NOPTS_VALUE) {
            m_currentFrameIndex = av_rescale_q(m_frame->pts, stream->time_base, av_inv_q(stream->avg_frame_rate));
        } else {
            m_currentFrameIndex++;
        }
        return true;
    }
    return false;
}

QSize FFmpegVideoSource::currentResolution() const {
    return m_resolution;
}

int64_t FFmpegVideoSource::frameCount() const {
    return m_totalFrames;
}

int64_t FFmpegVideoSource::currentFrameIndex() const {
    return m_currentFrameIndex;
}

double FFmpegVideoSource::nativeFps() const {
    return m_fps;
}

bool FFmpegVideoSource::skipFrame() {
    std::lock_guard<std::mutex> lock(m_mutex);
    cv::Mat dummy;
    return readFrameInternal(dummy);
}

bool FFmpegVideoSource::seekToFrame(int64_t frameIndex) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_formatCtx || m_videoStreamIndex < 0) return false;

    AVStream* stream = m_formatCtx->streams[m_videoStreamIndex];
    
    // Calculate target timestamp based on frame index and FPS
    int64_t targetTimestamp = av_rescale_q(frameIndex, 
                                           AVRational{1, (int)m_fps}, 
                                           stream->time_base);

    if (avformat_seek_file(m_formatCtx, m_videoStreamIndex, 
                           INT64_MIN, targetTimestamp, INT64_MAX, 
                           AVSEEK_FLAG_BACKWARD) < 0) {
        qDebug() << "[FFmpeg] Seek failed for frame:" << frameIndex << "targetTS:" << targetTimestamp;
        return false;
    }

    avcodec_flush_buffers(m_codecCtx);
    m_currentFrameIndex = -1; // Force re-sync

    // Accurate Seek: Decode frames until we reach the exact target frame
    cv::Mat dummy;
    int maxDiscards = 150; // GOP safety limit
    while (maxDiscards-- > 0) {
        if (!readFrameInternal(dummy)) break;
        if (m_currentFrameIndex >= frameIndex) break;
    }

    m_currentFrameIndex = frameIndex;
    return true;
}
