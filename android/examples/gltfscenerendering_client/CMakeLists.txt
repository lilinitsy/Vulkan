cmake_minimum_required(VERSION 3.4.1 FATAL_ERROR)

set(NAME gltfscenerendering_client)

set(SRC_DIR ../../../examples/${NAME})
set(BASE_DIR ../../../base)
set(EXTERNAL_DIR ../../../external)

include_directories(${BASE_DIR})
include_directories(${EXTERNAL_DIR})
include_directories(${EXTERNAL_DIR}/glm)
include_directories(${EXTERNAL_DIR}/imgui)
include_directories(${EXTERNAL_DIR}/tinygltf)
include_directories(${ANDROID_NDK}/sources/android/native_app_glue)
include_directories($ENV{ANDROID_FFMPEG}/include)
link_directories($ENV{ANDROID_FFMPEG}/lib)
include_directories($ENV{NONDROID_CLHPP}/include)
include_directories($ENV{NONDROID_CL_HEADERS}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -DVK_USE_PLATFORM_ANDROID_KHR -DVK_NO_PROTOTYPES")

file(GLOB EXAMPLE_SRC "${SRC_DIR}/*.cpp")

add_library(native-lib SHARED ${EXAMPLE_SRC})

add_library(native-app-glue STATIC ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)

add_subdirectory(../base ${CMAKE_SOURCE_DIR}/../base)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

target_link_libraries(
    native-lib
    native-app-glue
    libbase
    android
    log
    z
    avcodec
    avutil
    swresample
    # $ENV{ANDROID_FFMPEG}/lib/libavcodec.a
    # $ENV{ANDROID_FFMPEG}/lib/libavutil.a
    # $ENV{ANDROID_FFMPEG}/lib/libswresample.a
)
