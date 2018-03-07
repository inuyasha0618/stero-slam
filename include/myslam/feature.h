#ifndef FEATURE_H
#define FEATURE_H

#include "common_include.h"

namespace myslam {
    class MapPoint;

    class Feature {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Eigen::Vector2f pixel_ = Eigen::Vector2f(0, 0); //　该feature的像素坐标（注意！feature都是生存在图片里的）
        shared_ptr<MapPoint> mapPoint_ = nullptr; // 该feature关联的地图点
        float invDepth_ = -1;               // 逆深度
        bool isOutlier_ = false;
    };
}

#endif