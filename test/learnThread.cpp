#include <iostream>
#include <thread>
using namespace std;

void thread_function() {
    for (int i = 0; i < 30000; ++i) {
        cout << "thread function executing" << endl;
    }
}

int main(int argc, char** argv) {
    thread thread1(thread_function);

    for (int i = 0; i < 10000; ++i) {
        cout << "In main function" << endl;
    }

    thread1.join();
    cout << "thread 1 complete" << endl;
    return 0;
}
