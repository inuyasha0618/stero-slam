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

    void Backend::doLocalBA() {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
        g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        // 窗口内的关键帧设为g2o顶点
        for (shared_ptr<Frame> frame: kfWindow_) {
            g2o::VertexSE3Expmap* frameVertex = new g2o::VertexSE3Expmap();
            frameVertex->setId(frame->id_);
            frameVertex->setEstimate(g2o::SE3Quat(frame->T_c_w_.rotation_matrix(), frame->T_c_w_.translation()));
            // 窗口的第一帧是久经优化的了，所以当它成为窗口第一帧时是不需要优化的，也给其他帧提供了参照
            if (frame == kfWindow_.front()) {
                frameVertex->setFixed(true);
            }

            optimizer.addVertex(frameVertex);
        }
    }
}