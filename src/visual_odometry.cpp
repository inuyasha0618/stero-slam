#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include <boost/timer.hpp>
#include <g2o/core/block_solver.h>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"
#include "myslam/settings.h"

#define MIN_NUM_FEATURE 100

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

//                vector<cv::KeyPoint> kps;
//                for (auto feature: curr_->leftFeatures_) {
//                    cv::KeyPoint kp;
//                    kp.pt.x = feature->pixel_[0];
//                    kp.pt.y = feature->pixel_[1];
//                    kps.push_back(kp);
//                }
//
//                cv::Mat output;
//                cv::drawKeypoints(curr_->img_left_, kps, output);
//                cv::imshow("kps: ", output);
//                cv::waitKey(0);

                addStereoMapPoints(curr_);
                break;
            }
            case OK:
            {
                curr_ = frame;
                featrureTracking();
                poseEstimationPnP();
                int validFeatures = cleanBadFeatures();
                speed_ = curr_->T_c_w_ * ref_->T_c_w_.inverse();
                cout << "valid features: " << validFeatures << endl;

//                curr_->leftFeatures_.clear();
//                    featureDetection(curr_);
//                    stereoMatching(curr_);
//                    addStereoMapPoints(curr_);

                if (validFeatures < MIN_NUM_FEATURE) {
//                    cv::waitKey(0);
                    curr_->leftFeatures_.clear();
                    featureDetection(curr_);
                    stereoMatching(curr_);
                    addStereoMapPoints(curr_);
                }

                bool checkRes = checkEstimatedPose();
                cout << "checkRes: " << checkRes << endl;
                ref_ = curr_;
//                if (checkRes) {
//                    //　本帧就算弄完了，把它变成参考帧，供下个帧使用
//                    // 合格了，本帧就作为参考帧
//                    ref_ = curr_;
//                    num_lost_ = 0;
//
////                    if (checkKeyFrame()) {
////                        addKeyFrame();
////                    }
//                } else {
//                    num_lost_++;
//                    if (num_lost_ > max_num_lost_) {
//                        state_ = LOST;
//                    }
//                    return false;
//                }
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
            if (feature == nullptr)
                continue;
            ref_points.push_back(cv::Point2f(feature->pixel_[0], feature->pixel_[1]));
        }

        vector<uchar> status;
        vector<float> err;

        cv::calcOpticalFlowPyrLK(ref_->img_left_, curr_->img_left_, ref_points, curr_points, status, err);

        cv::findFundamentalMat(ref_points, curr_points, cv::FM_RANSAC, 3.0, 0.99, status);

        vector<cv::DMatch> matches;
        vector<cv::KeyPoint> kps_ref, kps_curr;

        size_t curr_idx = 0;
        for (size_t i = 0; i < curr_points.size(); i++) {

//            cv::KeyPoint kp_curr;
//            kp_curr.pt.x = curr_points[i].x;
//            kp_curr.pt.y = curr_points[i].y;
//            kps_curr.push_back(kp_curr);
//
//            cv::KeyPoint kp_ref;
//            kp_ref.pt.x = ref_points[i].x;
//            kp_ref.pt.y = ref_points[i].y;
//            kps_ref.push_back(kp_ref);

            if (status[i] && (curr_points[i].x > myslam::BORDER && curr_points[i].y > myslam::BORDER &&
                              curr_points[i].x < myslam::IMAGE_WIDTH - myslam::BORDER &&
                              curr_points[i].y < myslam::IMAGE_HEIGHT - myslam::BORDER)) {
                shared_ptr<Feature> feature(new Feature);
                feature->pixel_ = Eigen::Vector2d(curr_points[i].x, curr_points[i].y);
                if (ref_->leftFeatures_[i]->mapPoint_ != nullptr) {
                    feature->mapPoint_ = ref_->leftFeatures_[i]->mapPoint_;
                }

                feature->mapPoint_ = ref_->leftFeatures_[i]->mapPoint_;

                curr_->leftFeatures_.push_back(feature);

//                cv::DMatch match;
//
//                match.trainIdx = i;
//                match.queryIdx = i;
//                matches.push_back(match);
            }
        }
//        cv::Mat outputImg;
//        cv::drawMatches(ref_->img_left_,kps_ref, curr_->img_left_, kps_curr, matches, outputImg);
//        cv::imshow("feature tracking: ", outputImg);
//        cv::waitKey(0);

    }

    void VisualOdometry::poseEstimationPnP() {
        // 谜之delta
        const float delta = sqrt(5.991);
//        vector<cv::Point3f> pts3d;
//        vector<cv::Point2f> pts2d, pts2d_r; //　左右视图的像素坐标
//
//        for (auto feature: curr_->leftFeatures_) {
//            if (feature->mapPoint_ != nullptr) {
//                Eigen::Vector3d EigenPos3D = feature->mapPoint_->pos_;
//                Eigen::Vector2d EigenPixel = feature->pixel_;
//                cv::Point3f cvPos3D(EigenPos3D[0], EigenPos3D[1], EigenPos3D[2]);
//                pts3d.push_back(cvPos3D);
//                cv::Point2f cvPixel(EigenPixel[0], EigenPixel[1]);
//                pts2d.push_back(cvPixel);
//            }
//        }
//
//
//        cv::Mat K = (cv::Mat_<double>(3, 3)
//                << ref_->camera_->fx_, 0, ref_->camera_->cx_,
//                0, ref_->camera_->fy_, ref_->camera_->cy_,
//                0, 0, 1.0);
//        cv::Mat rvec, tvec, inliers;
//        cv::solvePnPRansac(pts3d, pts2d, K, cv::Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
//        num_inliers_ = inliers.rows;
//        curr_->T_c_w_ = Sophus::SE3(
//                Sophus::SO3(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0)),
//                Eigen::Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))
//        );

        // g2o初始化
        g2o::SparseOptimizer optimizer;
//        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> Block;
//        Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();

        g2o::BlockSolverX::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
        g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);


//        Block* solver_ptr = new Block(linearSolver);
//        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
        g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
        optimizer.setAlgorithm( solver );


        // 下面就是给优化问题添加顶点和边

        // 创建顶点,　该问题就一个顶点，即相机的相对参考帧的位姿
        g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
        pose->setId(0);
        pose->setFixed(false);
        // 设置优化初始值
        Sophus::SE3 T_c_w_esti = speed_ * ref_->T_c_w_;
        cout << T_c_w_esti.matrix() << endl;
        pose->setEstimate(g2o::SE3Quat(T_c_w_esti.rotation_matrix(), T_c_w_esti.translation()));

        // 将该点加入优化问题中
        optimizer.addVertex(pose);

        vector<EdgeProjXYZ2UVPoseOnly*> edges;
        vector<size_t> validFeatureIdx;

        // 添加边
        for (int i = 0; i < curr_->leftFeatures_.size(); i++) {
            // 创建边
            shared_ptr<Feature> &feature = curr_->leftFeatures_[i];
            if (!feature->mapPoint_)
                continue;
            EdgeProjXYZ2UVPoseOnly* edge = new EdgeProjXYZ2UVPoseOnly();
            edge->setId(i);

            edge->setVertex(0, pose);
            edge->point_ = feature->mapPoint_->pos_;
            edge->setMeasurement(feature->pixel_);
            edge->camera_ = curr_->camera_.get();

            edge->setInformation(Eigen::Matrix2d::Identity());

            // 由于误匹配的存在，要设置robust kernel
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
            edge->setRobustKernel( rk );
            rk->setDelta( delta );

            optimizer.addEdge(edge);
            edges.push_back(edge);
            validFeatureIdx.push_back(i);
        }


        //　开始优化
        optimizer.setVerbose(true);

//        optimizer.initializeOptimization(0);
//        optimizer.optimize(10);

        for (size_t it = 0; it < 4; it++) {
            pose->setEstimate(g2o::SE3Quat(T_c_w_esti.rotation_matrix(), T_c_w_esti.translation()));
            optimizer.initializeOptimization(0);
            optimizer.optimize(10);

            for (size_t i = 0; i < edges.size(); i++) {
                EdgeProjXYZ2UVPoseOnly* edge = edges[i];
                shared_ptr<Feature> feature = curr_->leftFeatures_[validFeatureIdx[i]];
                if (feature->isOutlier_) {
                    edge->computeError();
                }

                if (edge->chi2() > 5.991) {
                    feature->isOutlier_ = true;
                    edge->setLevel(1);
                } else {
                    feature->isOutlier_= false;
                    edge->setLevel(0);
                }

                if (it == 2) {
                    edge->setRobustKernel(0);
                }
            }
        }

        curr_->T_c_w_ = Sophus::SE3(pose->estimate().rotation(), pose->estimate().translation());

    }

    int VisualOdometry::cleanBadFeatures() {
        int valid_features = 0;
        for (size_t i = 0; i < curr_->leftFeatures_.size(); i++) {
            shared_ptr<Feature> feature = curr_->leftFeatures_[i];
            if (feature->isOutlier_ || feature->mapPoint_ == nullptr) {
                cout << "bad!!!" << endl;
                feature = nullptr;
                continue;
            }
            valid_features++;
        }
        return valid_features;
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
        clock_t start = clock();
        curr_->assignFeaturesToGrid();

        // Todo 应该将图片网格化，　然后每个格子里只提一个特征点
        // 从１开始提，可以有效避免提在边缘处
        for (int i = 1; i < FRAM_GRID_ROWS- 1; i++) {
            for (int j = 1; j < FRAM_GRID_COLS - 1; j++) {
                //　这样保证了只有没有提点的格子才提点，有点的格子就不管了, 加快提点的效率
                if (!curr_->frame_grid_[FRAM_GRID_COLS * i + j].empty()) {
                    cout << "不为空" << endl;
                    continue;
                }
                cout << "为空！" << endl;
                // 当前图像块
                cv::Mat block = curr_->img_left_(cv::Rect(j * FRAM_GRID_SIZE, i * FRAM_GRID_SIZE, FRAM_GRID_SIZE, FRAM_GRID_SIZE));
                vector<cv::KeyPoint> block_kps;
                cv::FAST(block, block_kps, 20, true);

                // Todo 找到本图像块中最好的特征点
                int x_start = j * FRAM_GRID_SIZE;
                int y_start = i * FRAM_GRID_SIZE;

                float scoreBest = -1;
                int idxBest = 0;
                for (size_t k = 0; k < block_kps.size(); k++) {
                    cv::KeyPoint &kp = block_kps[k];
                    kp.pt.x += x_start;
                    kp.pt.y += y_start;
                    float score = ShiTomasiScore(frame->img_left_, kp.pt.x, kp.pt.y);
                    if (score > scoreBest) {
                        scoreBest = score;
                        idxBest = k;
                    }

                }

                if (scoreBest < 0) continue;

                shared_ptr<Feature> feature(new Feature);
                feature->pixel_ = Eigen::Vector2d(block_kps[idxBest].pt.x, block_kps[idxBest].pt.y);
                frame->leftFeatures_.push_back(feature);
            }
        }

        clock_t end = clock();
        double time_cost = double(end - start) / CLOCKS_PER_SEC;
        cout << "feature detection cost time: " << time_cost << " seconds" << endl;

    }

    void VisualOdometry::stereoMatching(shared_ptr<Frame> frame) {
        clock_t start = clock();
        vector<cv::Point2f> leftPts, rightPts;
        vector<uchar> status;
        vector<float> err;
        for (auto feature: frame->leftFeatures_) {
            //　已经左右目匹配过的点就不用管了
            if (feature == nullptr || feature->mapPoint_ != nullptr)
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

        clock_t end = clock();
        double time_cost = double(end - start) / CLOCKS_PER_SEC;
        cout << "stereo matching cost time: " << time_cost << " seconds" << endl;
    }

    void VisualOdometry::addStereoMapPoints(shared_ptr<Frame> frame) {
        for (auto feature: frame->leftFeatures_) {
            // 关联３D地图点的就不用新建３D点了
            if (feature->mapPoint_ != nullptr)
                continue;

            // 是新提的点，但是深度无效，那么也没办法，无法创建有效的3D点
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