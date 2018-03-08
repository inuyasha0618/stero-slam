add_executable( run_vo run_vo.cpp )
target_link_libraries( run_vo myslam )

add_executable(testFast testFast.cpp)
target_link_libraries(testFast ${OpenCV_LIBS})

add_executable(testBlock testBlock.cpp)
target_link_libraries(testBlock ${OpenCV_LIBS})

add_executable(testSteroMatching testSteroMatching.cpp)
target_link_libraries(testSteroMatching myslam)

add_executable(testSteroFullLK testSteroFullLK.cpp)
target_link_libraries(testSteroFullLK myslam)

add_executable(testLKMatching testLKMatching.cpp)
target_link_libraries(testLKMatching myslam)

add_executable(testSteroLKMatching testSteroLKMatching.cpp)
target_link_libraries(testSteroLKMatching myslam)

add_executable(learnThread learnThread.cpp)