add_executable( run_vo run_vo.cpp )
target_link_libraries( run_vo myslam )

add_executable(testSteroMatching testSteroMatching.cpp)
target_link_libraries(testSteroMatching myslam)

add_executable(learnThread learnThread.cpp)