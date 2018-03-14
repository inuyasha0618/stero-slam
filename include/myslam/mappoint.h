#ifndef MAP_POINT_H
#define MAP_POINT_H

#include "common_include.h"

namespace myslam
{
    class Frame;

    // Todo 由于mappoint被前后端所共享，所以应该加些锁
    class MapPoint
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<MapPoint> Ptr;
        unsigned long id_;
        Eigen::Vector3d pos_;     // 该地图点的世界坐标
        Eigen::Vector3d norm_;
        cv::Mat descriptor_;
        int observed_times_;
        int correct_times_;

        mutex mutexPointPos_;

        MapPoint();
        MapPoint(unsigned long id, Eigen::Vector3d pos, Eigen::Vector3d norm);

        static MapPoint::Ptr createMapPoint();

        weak_ptr<Frame> refKF_; // 第一次观测到此地图点的关键帧

        // 维护了一个map, key是观测到该点的帧，以及在该帧当中是第几个feature
        typedef std::map<weak_ptr<Frame>, size_t, std::owner_less<weak_ptr<Frame>>> ObsMap;
        ObsMap observations_;

        void addObservation(shared_ptr<Frame> keyFrame, size_t index);

        Eigen::Vector3d getWorldPos() {
            // Todo 看看是否需要加锁
            unique_lock<mutex> lock(mutexPointPos_);
            return pos_;
        }

        void setWorldPos(Eigen::Vector3d pos) {
            unique_lock<mutex> lock(mutexPointPos_);
            pos_ = pos;
        }

        // Todo updateWorldPos 用在当初始观测到该地图点的关键帧expired掉之后，地图点关联到新的关键帧之后，重新三角化出一个新的坐标
        // 再想想是否有用，因为地图点的3D坐标会被后端优化，如果用这样简简单单的通过三角化更新其坐标的话，是否失去了后端优化的好处了
        void updateWorldPos();
    };
}

#endif