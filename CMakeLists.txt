cmake_minimum_required(VERSION 2.8)
project(segformer_demo)

add_definitions(-std=c++11)
add_definitions("-Wall")
add_definitions(-DUNICODE -D_UNICODE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/common)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

find_package( OpenCV REQUIRED )

if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "opencv Not Found!")
else()
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
endif (NOT OpenCV_FOUND)


set(SRC
        src/demo.cpp
        include/common/tengine_operations.c
        )

add_executable(segformer_demo ${SRC})

target_link_libraries(segformer_demo tengine-lite ${OpenCV_LIBS})
