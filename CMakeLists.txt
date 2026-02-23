cmake_minimum_required(VERSION 3.24)
project(pathsig_ops LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Torch REQUIRED)

add_library(pathsig_ops SHARED
        src/pathsig/bindings.cpp
        src/pathsig/forward/sig_kernels.cu
        src/pathsig/forward/sig_launch.cu
        src/pathsig/backward/sig_backward_kernels.cu
        src/pathsig/backward/sig_backward_launch.cu
)

target_include_directories(pathsig_ops PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pathsig
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pathsig/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pathsig/forward
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pathsig/backward
)

if (TARGET Torch::Torch)
    target_link_libraries(pathsig_ops PRIVATE Torch::Torch)
else()
    target_link_libraries(pathsig_ops PRIVATE ${TORCH_LIBRARIES})
endif()

separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(pathsig_ops PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS_LIST}>
        $<$<COMPILE_LANGUAGE:CUDA>:${TORCH_CXX_FLAGS_LIST}>
)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

set_target_properties(pathsig_ops PROPERTIES
        OUTPUT_NAME "pathsig_ops"
        # PREFIX "lib"  # optional; redundant on Linux
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if (UNIX AND NOT APPLE)
    set_target_properties(pathsig_ops PROPERTIES
            INSTALL_RPATH "$ORIGIN"
    )
endif()

install(TARGETS pathsig_ops
        LIBRARY DESTINATION pathsig
        RUNTIME DESTINATION pathsig
        ARCHIVE DESTINATION pathsig
)
