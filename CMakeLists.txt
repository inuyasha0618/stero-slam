cmake_minimum_required(VERSION 2.8)
project(stero-slam)

SET(CMAKE_CXX_COMPILER "g++")
SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS "-std=c++11 -pthread")

SET(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

#Eigen
include_directories("/usr/include/eigen3")

#OpenCV
FIND_PACKAGE(OpenCV 3 REQUIRED)

#Sophus
set(SOPHUS_LIBS /home/slam/Sophus/libSophus.so)
include_directories(/home/slam/Sophus)

#G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

find_package( Cholmod )
include_directories( ${CHOLMOD_INCLUDE_DIR} )

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${SOPHUS_LIBS}
        ${CHOLMOD_LIBRARIES}
#        g2o_core g2o_stuff g2o_types_sba
        g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sim3 g2o_types_sba
        )

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/src)
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/test)

