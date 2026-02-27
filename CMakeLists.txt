cmake_minimum_required(VERSION 3.24)
project(pathsig_ops LANGUAGES CXX CUDA)

# Standards & defaults
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CUDA_STANDARD         17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)
set(CMAKE_CUDA_EXTENSIONS       OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET    hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# NVRTC workaround for some CUDA 13.x installs that only ship versioned libnvrtc.so.<ver> with no
# unversioned symlink.
function(_pathsig_find_nvrtc out)
    set(roots "")
    foreach(env CUDA_HOME CUDA_PATH)
        if(DEFINED ENV{${env}})
            list(APPEND roots "$ENV{${env}}")
        endif()
    endforeach()
    if(CMAKE_CUDA_COMPILER)
        get_filename_component(_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
        get_filename_component(_root "${_dir}/.." REALPATH)
        list(APPEND roots "${_root}")
    endif()
    list(APPEND roots "/usr/local/cuda")
    list(REMOVE_DUPLICATES roots)

    foreach(root IN LISTS roots)
        foreach(libdir
                "${root}/lib64"
                "${root}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib"
                "${root}/targets/x86_64-linux/lib"
        )
            if(EXISTS "${libdir}/libnvrtc.so")
                set(${out} "${libdir}/libnvrtc.so" PARENT_SCOPE)
                return()
            endif()
        endforeach()

        file(GLOB _cands
                "${root}/lib64/libnvrtc.so.*"
                "${root}/targets/*/lib/libnvrtc.so.*"
        )
        if(_cands)
            list(SORT _cands COMPARE NATURAL ORDER DESCENDING)
            list(GET  _cands 0 _best)
            set(${out} "${_best}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
endfunction()

_pathsig_find_nvrtc(_NVRTC)
if(_NVRTC)
    message(STATUS "NVRTC workaround: ${_NVRTC}")
    set(CUDA_nvrtc_LIBRARY "${_NVRTC}" CACHE FILEPATH "NVRTC library" FORCE)
else()
    message(WARNING "Could not locate NVRTC library; CUDA_nvrtc_LIBRARY left unset.")
endif()

# Torch
find_package(Torch REQUIRED)

if(DEFINED TORCH_CUDA_ARCH_LIST)
    message(STATUS "TORCH_CUDA_ARCH_LIST = ${TORCH_CUDA_ARCH_LIST}")
elseif(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    message(STATUS "TORCH_CUDA_ARCH_LIST (env) = $ENV{TORCH_CUDA_ARCH_LIST}")
endif()

# Library target
add_library(pathsig_ops SHARED
        src/pathsig/bindings.cpp
        src/pathsig/forward/sig_kernels.cu
        src/pathsig/forward/sig_launch.cu
        src/pathsig/backward/sig_backward_kernels.cu
        src/pathsig/backward/sig_backward_launch.cu
)

target_include_directories(pathsig_ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pathsig
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pathsig/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pathsig/forward
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pathsig/backward
)

if(TARGET Torch::Torch)
    target_link_libraries(pathsig_ops PRIVATE Torch::Torch)
else()
    target_link_libraries(pathsig_ops PRIVATE ${TORCH_LIBRARIES})
    target_include_directories(pathsig_ops PRIVATE ${TORCH_INCLUDE_DIRS})
endif()

string(REGEX MATCH "-D_GLIBCXX_USE_CXX11_ABI=([01])" _abi_match "${TORCH_CXX_FLAGS}")
if(_abi_match)
    target_compile_definitions(pathsig_ops PRIVATE _GLIBCXX_USE_CXX11_ABI=${CMAKE_MATCH_1})
endif()

set_target_properties(pathsig_ops PROPERTIES
        OUTPUT_NAME                pathsig_ops
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(UNIX AND NOT APPLE)
    set_target_properties(pathsig_ops PROPERTIES
            BUILD_RPATH_USE_ORIGIN ON
            INSTALL_RPATH          "$ORIGIN"
    )
endif()

# Install
install(TARGETS pathsig_ops
        LIBRARY DESTINATION pathsig
        RUNTIME DESTINATION pathsig
        ARCHIVE DESTINATION pathsig
)