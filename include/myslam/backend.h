#ifndef BACKEND_H
#define BACKEND_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"
#include "myslam/g2o_types.h"

namespace myslam {
    class Frame;
    class VisualOdometry;
    class MapPoint;

    // Todo 由于backend会被两个现成所共享，所以应该设置一些锁
    class Backend {
    public:
        Backend(shared_ptr<VisualOdometry> visualOdometry): vo_(visualOdometry) {
            backendThread = thread(&Backend::MainLoop, this);
        }
        void MainLoop();
        void processNewKeyFrame();
        void insertKeyFrame(shared_ptr<Frame> frame);
    private:
        shared_ptr<VisualOdometry> vo_ = nullptr;
        set<shared_ptr<MapPoint>> localMap_;
        deque<shared_ptr<Frame>> kfWindow_;
        deque<shared_ptr<Frame>> newKfs_;

        thread backendThread;
        condition_variable cond_;
        mutex mutexNewKfs;
        shared_ptr<Frame> currFrame_; // 正在处理的帧（也可理解为最新的一帧）
        void doLocalBA(); // 局部地图 + 窗口内的关键帧做一次Bundle Adjustment

        void cleanMapPoints();
    };
}
#endif