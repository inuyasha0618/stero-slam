#include "myslam/backend.h"

namespace myslam {
    void Backend::MainLoop() {
        while(1) {
            unique_lock<mutex> lock(mutexNewKfs);
            cond_.wait(lock, [&](){return !this->newKfs_.empty();});
            currFrame_ = newKfs_.front();
            newKfs_.pop_front();
            kfWindow_.push_back(currFrame_);
            lock.unlock();

            processNewKeyFrame();
        }
    }

    void Backend::insertKeyFrame(shared_ptr<Frame> frame) {
        unique_lock<mutex> lock(mutexNewKfs);
        newKfs_.push_back(frame);
        cond_.notify_all();
    }

    void Backend::processNewKeyFrame() {
        cout << "Backend线程收到一个新的关键帧，　对应图片为 " << currFrame_->imgPath << endl;

        for (size_t i = 0; i < currFrame_->leftFeatures_.size(); i++) {
            shared_ptr<Feature> feature = currFrame_->leftFeatures_[i];

            // 该feature没有对应的mapPoint则继续
            if (feature->mapPoint_ == nullptr) {
                continue;
            }

            // 有mapPoint，则看看局部地图里有这个点没，没有就加进去，有了就拉倒
            if (localMap_.count(feature->mapPoint_) == 0) {
                localMap_.insert(feature->mapPoint_);
                feature->mapPoint_->addObservation(currFrame_, i);
            }
        }

        // 新来的关键帧处理完毕(即新的地图点加到局部地图中去)，开始做localBA
        doLocalBA();

        // 保持窗口关键帧的数量
        if (kfWindow_.size() > 10) {
            kfWindow_.pop_front(); // 把最早的那个关键帧删掉
        }
    }

    void Backend::cleanMapPoints() {

    }

    void Backend::doLocalBA() {

        const float delta = sqrt(5.991);
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
        g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

//        map<int, string> map1;
        // Todo 看看其他地方的vector需要reserve开辟一下空间不？

        // 窗口内的关键帧设为g2o顶点, 并添加进去
        int maxFrameVertexId = 0; // 帮助下面的地图点vertex往下排序号
        for (int i = 0; i < kfWindow_.size(); i++) {
            shared_ptr<Frame> frame = kfWindow_[i];
            g2o::VertexSE3Expmap* frameVertex = new g2o::VertexSE3Expmap();
            frameVertex->setId(frame->id_);

            if (frame->id_ > maxFrameVertexId) {
                maxFrameVertexId = frame->id_;
            }

            frameVertex->setEstimate(g2o::SE3Quat(frame->T_c_w_.rotation_matrix(), frame->T_c_w_.translation()));
            // 窗口的第一帧是久经优化的了，所以当它成为窗口第一帧时是不需要优化的，也给其他帧提供了参照
            if (i == 0) {
                frameVertex->setFixed(true);
            }

            optimizer.addVertex(frameVertex);
        }

        // 添加地图点pos顶点
        int mpIdx = 0;
        vector<int> validPointVertexIdx;
        vector<shared_ptr<MapPoint>> validPoints;

        validPointVertexIdx.reserve(localMap_.size());
        validPoints.reserve(localMap_.size());

        for (shared_ptr<MapPoint> mp: localMap_) {
            if (mp == nullptr) {
                continue;
            }

            g2o::VertexSBAPointXYZ* pointVertex = new g2o::VertexSBAPointXYZ();

            int currIdx = maxFrameVertexId + 1 + mpIdx;

            pointVertex->setId(currIdx);
            mpIdx++;
            pointVertex->setEstimate(mp->getWorldPos());
            pointVertex->setMarginalized(true);

            // Todo 把被观测到两次及以上的地图点才加入优化，当前先把所有地图点都加入优化
            // 不能急着加入该点，因为该点可能已经不能被窗口内的帧所观测到了
//            optimizer.addVertex(pointVertex);

            // 依据该点的观测添加边
            // 该点被观测到的次数
            int observedTimes = 0, edgeId = 0;
            for (auto &obs: mp->observations_) {
                auto f = obs.first;
                if (f.expired()) {
                    continue;
                }

                if (observedTimes == 0) {
                    optimizer.addVertex(pointVertex);

                    validPoints.push_back(mp);
                    validPointVertexIdx.push_back(currIdx);
                }

                observedTimes++;
                shared_ptr<Frame> frame = f.lock();

                // 找到在该帧当中相应的feature
                shared_ptr<Feature> relativeFeat = frame->leftFeatures_[obs.second];

                // 创建一条边
                EdgeProjXYZ2UV* edge = new EdgeProjXYZ2UV();
                edge->setId(edgeId++);
                edge->setVertex(0, pointVertex);
                edge->setVertex(1, optimizer.vertex(frame->id_));
                edge->setMeasurement(relativeFeat->pixel_);
                edge->camera_ = frame->camera_.get();
                edge->setInformation(Eigen::Matrix2d::Identity());

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                edge->setRobustKernel( rk );
                rk->setDelta( delta );
                optimizer.addEdge(edge);
            }

            // 如果改点在窗口帧当中有被观测到，则添加这个顶点，否则删除之
            if (observedTimes == 0) {
                delete pointVertex;
            }
        }

        optimizer.initializeOptimization(0);
        optimizer.optimize(5);

        // Todo 优化完毕， 需要根据优化的结果更新相应的关键帧及地图点的坐标
        // 更新窗口内的关键帧的位姿
        for (shared_ptr<Frame> frame: kfWindow_) {
            g2o::VertexSE3Expmap* frameVertex = (g2o::VertexSE3Expmap*)optimizer.vertex(frame->id_);
            Sophus::SE3 Tcw = Sophus::SE3(frameVertex->estimate().rotation(), frameVertex->estimate().translation());
            frame->setPose(Tcw);
        }

        // Todo 更新局部地图地图点的世界坐标
        for (int i = 0; i < validPoints.size(); i++) {
            shared_ptr<MapPoint> mp = validPoints[i];
            int vertexIdx = validPointVertexIdx[i];

            g2o::VertexSBAPointXYZ* pointVertex = (g2o::VertexSBAPointXYZ*)optimizer.vertex(vertexIdx);
            mp->setWorldPos(pointVertex->estimate());
        }
    }
}