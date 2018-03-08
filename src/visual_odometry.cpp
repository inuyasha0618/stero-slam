#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include <boost/timer.hpp>
#include <g2o/core/block_solver.h>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"

#define MIN_NUM_FEATURE 2000

namespace myslam
{
    VisualOdometry::VisualOdometry():
    state_(INITIALIZING), map_(new Map), ref_(nullptr), curr_(nullptr), num_lost_(0), num_inliers_(0)
    {
        num_of_features_ = Config::getParam<int>("number_of_features");
        scale_factor_ = Config::getParam<double>("scale_factor");
        level_pyramid_ = Config::getParam<int>("level_pyramid");
        match_ratio_ = Config::getParam<float>("match_ratio");
        max_num_lost_ = Config::getParam<int>("max_num_lost");
        min_inliers_ = Config::getParam<int>("min_inliers");
        key_frame_min_rot_ = Config::getParam<double>("key_frame_min_rot");
        key_frame_min_trans_ = Config::getParam<double>("key_frame_min_trans");

        orb_ = cv::ORB::create(num_of_features_, scale_factor_, level_pyramid_);
    }

    VisualOdometry::~VisualOdometry() {}

    bool VisualOdometry::addFrame(Frame::Ptr frame) {
        switch (state_) {
            case INITIALIZING:
            {
                state_ = OK;
                curr_ = ref_ = frame;
                map_->insertKeyFrame(frame);
                featureDetection(curr_);
                stereoMatching(curr_);
                addStereoMapPoints(curr_);
                break;
            }
            case OK:
            {
                curr_ = frame;
                featrureTracking();
                poseEstimationPnP();
                cout << "num inliers: " << num_inliers_ << endl;

                if (curr_->leftFeatures_.size() < MIN_NUM_FEATURE) {
                    curr_->leftFeatures_.clear();
                    featureDetection(curr_);
                    stereoMatching(curr_);
                    addStereoMapPoints(curr_);
                }


                if (checkEstimatedPose()) {
                    //　本帧就算弄完了，把它变成参考帧，供下个帧使用
                    // 合格了，本帧就作为参考帧
                    ref_ = curr_;
                    num_lost_ = 0;

//                    if (checkKeyFrame()) {
//                        addKeyFrame();
//                    }
                } else {
                    num_lost_++;
                    if (num_lost_ > max_num_lost_) {
                        state_ = LOST;
                    }
                    return false;
                }
                break;
            }
            case LOST:
            {
                break;
            }
        }
        return true;
    }

    void VisualOdometry::featrureTracking() {

        vector<cv::Point2f> ref_points, curr_points;

        for (auto feature: ref_->leftFeatures_) {
            ref_points.push_back(cv::Point2f(feature->pixel_[0], feature->pixel_[1]));
        }

        vector<uchar> status;
        vector<float> err;

        cv::calcOpticalFlowPyrLK(ref_->img_left_, curr_->img_left_, ref_points, curr_points, status, err);

        for (size_t i = 0; i < curr_points.size(); i++) {
            if (status[i]) {
                shared_ptr<Feature> feature(new Feature);
                feature->pixel_ = Eigen::Vector2d(curr_points[i].x, curr_points[i].y);
                if (ref_->leftFeatures_[i]->mapPoint_ != nullptr) {
                    feature->mapPoint_ = ref_->leftFeatures_[i]->mapPoint_;
                }
                curr_->leftFeatures_.push_back(feature);
            }
        }

    }

    void VisualOdometry::poseEstimationPnP() {
        // 谜之delta
        const float delta = sqrt(5.991);
        vector<cv::Point3f> pts3d;
        vector<cv::Point2f> pts2d, pts2d_r; //　左右视图的像素坐标

        for (auto feature: curr_->leftFeatures_) {
            if (feature->mapPoint_ != nullptr) {
                Eigen::Vector3d EigenPos3D = feature->mapPoint_->pos_;
                Eigen::Vector2d EigenPixel = feature->pixel_;
                cv::Point3f cvPos3D(EigenPos3D[0], EigenPos3D[1], EigenPos3D[2]);
                pts3d.push_back(cvPos3D);
                cv::Point2f cvPixel(EigenPixel[0], EigenPixel[1]);
                pts2d.push_back(cvPixel);
            }
        }


        cv::Mat K = (cv::Mat_<double>(3, 3)
                << ref_->camera_->fx_, 0, ref_->camera_->cx_,
                0, ref_->camera_->fy_, ref_->camera_->cy_,
                0, 0, 1.0);
        cv::Mat rvec, tvec, inliers;
        cv::solvePnPRansac(pts3d, pts2d, K, cv::Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
        num_inliers_ = inliers.rows;
        curr_->T_c_w_ = Sophus::SE3(
                Sophus::SO3(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0)),
                Eigen::Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))
        );

        // g2o初始化
//        g2o::SparseOptimizer optimizer;
//
//        g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType> ();
//        g2o::BlockSolver_6_3*   solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
//        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
//        optimizer.setAlgorithm( solver );
//
//
//
//        // 下面就是给优化问题添加顶点和边
//
//        // 创建顶点,　该问题就一个顶点，即相机的相对参考帧的位姿
//        g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
//        pose->setId(0);
//        pose->setFixed(false);
//        // 设置优化初始值
//        pose->setEstimate(g2o::SE3Quat(T_c_r_esti_.rotation_matrix(), T_c_r_esti_.translation()));
//
//        // 将该点加入优化问题中
//        optimizer.addVertex(pose);
//
//        // 添加边
//        for (int i = 0; i < inliers.rows; i++) {
//            // 创建边
//            int index = inliers.at<int>(i, 0);
//
//            EdgeProjXYZ2SteroUVPoseOnly* edge = new EdgeProjXYZ2SteroUVPoseOnly();
//            edge->setId(i);
//            edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
//
////            edge->setVertex(0, pose);
//
//            edge->setMeasurement(Eigen::Vector3d(pts2d[index].x, pts2d[index].y, pts2d_r[index].x));
////            edge->camera_ = curr_->camera_.get();
//            edge->fx = curr_->camera_->fx_;
//            edge->fy = curr_->camera_->fy_;
//            edge->cx = curr_->camera_->cx_;
//            edge->cy = curr_->camera_->cy_;
//            edge->bw = curr_->camera_->base_line_;
//            edge->point_ = Eigen::Vector3d(pts3d[index].x, pts3d[index].y, pts3d[index].z);
//            edge->setInformation(Eigen::Matrix3d::Identity());
//
//            // 由于误匹配的存在，要设置robust kernel
////            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
////            edge->setRobustKernel( rk );
////            rk->setDelta( delta );
//
//            optimizer.addEdge(edge);
//
//        }
//
//
//        //　开始优化
//        optimizer.setVerbose(true);
//        cout << "edges: " << optimizer.edges().size() << endl;
//        optimizer.initializeOptimization(0);
//        optimizer.optimize(10);
//
//        T_c_r_esti_ = Sophus::SE3(pose->estimate().rotation(), pose->estimate().translation());

    }

    bool VisualOdometry::checkEstimatedPose() {
        if (num_inliers_ < min_inliers_) {
            return false;
        }

        T_c_r_esti_ = curr_->T_c_w_ * ref_->T_c_w_.inverse();

        Sophus::Vector6d tcr_vec = T_c_r_esti_.log();
        if (tcr_vec.norm() > 5.0) {
            return false;
        }

        return true;
    }

    bool VisualOdometry::checkKeyFrame() {
        Sophus::Vector6d tcr_vec = T_c_r_esti_.log();
        Eigen::Vector3d trans = tcr_vec.head<3>();
        Eigen::Vector3d rot = tcr_vec.tail<3>();

        if (trans.norm() < key_frame_min_trans_ && rot.norm() < key_frame_min_rot_) {
            return false;
        }

        return true;
    }

    void VisualOdometry::addKeyFrame() {
        map_->insertKeyFrame(curr_);
    }

    void VisualOdometry::featureDetection(shared_ptr<Frame> frame) {
        vector<cv::KeyPoint> kps;
        cv::FAST(frame->img_left_, kps, 20, true);
        for (int i = 0; i < kps.size(); i++) {
            shared_ptr<Feature> feature(new Feature);
            feature->pixel_ = Eigen::Vector2d(kps[i].pt.x, kps[i].pt.y);
            frame->leftFeatures_.push_back(feature);
        }
    }

    void VisualOdometry::stereoMatching(shared_ptr<Frame> frame) {
        vector<cv::Point2f> leftPts, rightPts;
        vector<uchar> status;
        vector<float> err;
        for (auto feature: frame->leftFeatures_) {
            if (feature->mapPoint_ != nullptr)
                continue;
            leftPts.push_back(cv::Point2f(feature->pixel_[0], feature->pixel_[1]));
        }
        cv::calcOpticalFlowPyrLK(frame->img_left_, frame->img_right_, leftPts, rightPts, status, err);

        for (size_t i = 0; i < rightPts.size(); i++) {
            if (status[i]) {
                cv::Point2f& pl = leftPts[i];
                cv::Point2f& pr = rightPts[i];

                if (pl.x < pr.x || (fabs(pl.y - pr.y) > 3))
                    continue;
                float disparity = pl.x - pr.x;
                frame->leftFeatures_[i]->invDepth_ = disparity / (frame->camera_->fx_ * frame->camera_->base_line_);
            }
        }
    }

    void VisualOdometry::addStereoMapPoints(shared_ptr<Frame> frame) {
        for (auto feature: frame->leftFeatures_) {
            if (feature->invDepth_ == -1)
                continue;

            shared_ptr<MapPoint> mapPoint(new MapPoint);
            float depth = 1.0 / feature->invDepth_;
            Eigen::Vector3d Pc = frame->camera_->pixel2camera(feature->pixel_, depth);
            mapPoint->pos_ =  frame->T_c_w_.inverse() * Pc;

            // 将mapPoint与feature关联上
            feature->mapPoint_ = mapPoint;
        }
    }

}