# setup.py for CUDA Torch extension with correct ABI passing to NVCC host compiler
import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

current_dir = os.path.dirname(os.path.abspath(__file__))
abi_flag = '-D_GLIBCXX_USE_CXX11_ABI=' + ('1' if torch._C._GLIBCXX_USE_CXX11_ABI else '0')

setup(
    name='pathsig',
    version='0.1.0',
    package_dir={'': 'src'},
    packages=['pathsig'],
    ext_modules=[
        CUDAExtension(
            name='pathsig._impl',
            sources=[
                'src/pathsig/pybindings.cpp',
                'src/pathsig/utils/SigDecomposition.cpp',
                'src/pathsig/signature_backward/sig_backprop.cu',
                'src/pathsig/signature_backward/sig_backprop_launch.cu',
                'src/pathsig/signature_forward/compute_sig.cu',
                'src/pathsig/signature_forward/compute_sig_launch.cu',
                'src/pathsig/utils/sig_setup.cu',
            ],
            include_dirs=[
                os.path.join(current_dir, 'src/pathsig'),
                os.path.join(current_dir, 'src/pathsig/utils'),
                os.path.join(current_dir, 'src/pathsig/signature_backward'),
                os.path.join(current_dir, 'src/pathsig/signature_forward'),
            ],
            extra_compile_args={
                'cxx': ['-O2', abi_flag, '-fvisibility=hidden'],
                'nvcc': ['-O2', abi_flag, '-diag-suppress=20281'],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)