import os
import subprocess

import torch
from torch.utils import cpp_extension

from setuptools import find_namespace_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def corex_build_macros():
    """Use the same platform/architecture gates as cmake-corex-template."""
    if not getattr(torch, "corex", False):
        return []
    arch = getattr(torch, "corex_arch", None)
    targets = {"ivcore11": "BUILD_ON_MR", "ivcore20": "BUILD_ON_QS",
               "ivcore30": "BUILD_ON_CQ"}
    requested = os.environ.get("CMAKE_CUDA_ARCHITECTURES", arch)
    if requested not in targets:
        raise RuntimeError(f"Unsupported Corex target: {requested}")
    if requested != arch:
        raise RuntimeError(f"Corex target {requested} does not match PyTorch target {arch}")
    if arch == "ivcore30":
        # This SDK injects --skip-double even for CQ. Preserve CQ FP64.
        flags = cpp_extension.COMMON_COREX_FLAGS
        while "--skip-double" in flags:
            i = flags.index("--skip-double")
            if i and flags[i - 1] == "-mllvm":
                del flags[i - 1:i + 1]
            else:
                del flags[i]
    return [("BUILD_ON_ILUVATAR", "1"), (targets[arch], "1")]


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources],
        define_macros=corex_build_macros()
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    version = '0.6.0+%s' % get_git_commit_number()
    write_version_to_file(version, 'pcdet/version.py')

    setup(
        name='pcdet',
        version=version,
        description='OpenPCDet is a general codebase for 3D object detection from point cloud',
        install_requires=[
            'numpy',
            'llvmlite',
            'numba',
            'tensorboardX',
            'easydict',
            'pyyaml',
            'scikit-image',
            'tqdm',
            'SharedArray',
            'av2==0.3.6',
            # 'spconv',  # spconv has different names depending on the cuda version
        ],

        author='Shaoshuai Shi',
        author_email='shaoshuaics@gmail.com',
        license='Apache License 2.0',
        packages=find_namespace_packages(include=['pcdet*']),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='pcdet.ops.iou3d_nms',
                sources=[
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roiaware_pool3d_cuda',
                module='pcdet.ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                    'src/roiaware_pool3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roipoint_pool3d_cuda',
                module='pcdet.ops.roipoint_pool3d',
                sources=[
                    'src/roipoint_pool3d.cpp',
                    'src/roipoint_pool3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='pointnet2_stack_cuda',
                module='pcdet.ops.pointnet2.pointnet2_stack',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu', 
                    'src/interpolate.cpp', 
                    'src/interpolate_gpu.cu',
                    'src/voxel_query.cpp', 
                    'src/voxel_query_gpu.cu',
                    'src/vector_pool.cpp',
                    'src/vector_pool_gpu.cu'
                ],
            ),
            make_cuda_ext(
                name='pointnet2_batch_cuda',
                module='pcdet.ops.pointnet2.pointnet2_batch',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/interpolate.cpp',
                    'src/interpolate_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu',

                ],
            ),
            make_cuda_ext(
                name="bev_pool_ext",
                module="pcdet.ops.bev_pool",
                sources=[
                    "src/bev_pool.cpp",
                    "src/bev_pool_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name='ingroup_inds_cuda',
                module='pcdet.ops.ingroup_inds',
                sources=[
                    'src/ingroup_inds.cpp',
                    'src/ingroup_inds_kernel.cu',
                ]
            ),
        ],
    )
