#include "myslam/backend.h"

namespace myslam {
    void Backend::MainLoop() {
        while(1) {
            unique_lock<mutex> lock(mutexNewKfs);
            cond_.wait(lock, [](){return !newKfs_.empty();});
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
        cout << "收到一个新的关键帧，　对应图片为" << endl;
    }

    void Backend::doLocalBA() {
        
    }
}