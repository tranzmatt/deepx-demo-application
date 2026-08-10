cmake_minimum_required(VERSION 3.14)

project(CppQtUnifiedDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

# OpenCV
find_package(OpenCV REQUIRED)

# DXRT
find_library(DXRT_LIB NAMES dxrt PATHS /usr/local/lib /usr/lib)
if(NOT DXRT_LIB)
    # Without this the link line degenerates to -lDXRT_LIB-NOTFOUND
    message(FATAL_ERROR "DXRT library not found. Install dx_rt (expected libdxrt.so in /usr/local/lib).")
endif()
message(STATUS "DXRT library found: ${DXRT_LIB}")

# Raspberry Pi libcamera support (optional)
option(ENABLE_LIBCAMERA "Enable native Raspberry Pi camera support via libcamera" ON)
if(ENABLE_LIBCAMERA)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(LIBCAMERA libcamera)
    endif()
    if(LIBCAMERA_FOUND)
        message(STATUS "libcamera found: ${LIBCAMERA_LIBRARIES}")
        add_definitions(-DHAVE_LIBCAMERA)
    else()
        message(STATUS "libcamera not found - Raspberry Pi camera support disabled")
    endif()
endif()

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.h")

add_executable(cpp_qt_app ${SOURCES})

target_link_libraries(cpp_qt_app PRIVATE 
    Qt${QT_VERSION_MAJOR}::Widgets 
    ${OpenCV_LIBS}
    ${DXRT_LIB}
    pthread
)

# Link libcamera if available
if(LIBCAMERA_FOUND)
    target_include_directories(cpp_qt_app PRIVATE ${LIBCAMERA_INCLUDE_DIRS})
    target_link_libraries(cpp_qt_app PRIVATE ${LIBCAMERA_LIBRARIES})
endif()

target_include_directories(cpp_qt_app PRIVATE
    src
    src/core
    src/interfaces
    src/adapters
    src/ui
    src/core/postprocess
)

if(UNIX)
    target_link_options(cpp_qt_app PRIVATE -Wl,--no-as-needed)
endif()
