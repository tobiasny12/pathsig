# setup.py for CUDA Torch extension with correct ABI passing to NVCC host compiler
import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9;9.0;10.0;12.0+PTX'

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
                'cxx': ['-std=c++17', '-O3'],
                'nvcc': ['-O3'],
            },
            depends=['src/pathsig/utils/SigDecomposition.h',
                     'src/pathsig/utils/word_mappings.cuh',
                     'src/pathsig/utils/extended_precision.cuh',
                     'src/pathsig/utils/sig_setup.cuh',
                     'src/pathsig/signature_backward/sig_backprop_launch.cuh',
                     'src/pathsig/signature_forward/compute_sig_launch.cuh',
                     ]
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)