add_library(myslam SHARED
        camera.cpp
        frame.cpp
        mappoint.cpp
        map.cpp
        config.cpp
        visual_odometry.cpp
        g2o_types.cpp

        )

target_link_libraries(myslam ${THIRD_PARTY_LIBS})

ADD_EXECUTABLE(hello hello.cpp)