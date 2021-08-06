import os
import glob
from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_extensions():
    extensions = []
    ext_name = '_ext'
    op_files = glob.glob('./layers/csrc/*')
    print(op_files)
    include_path = os.path.abspath('./layers/cinclude')

    extensions.append(CUDAExtension(
        name=ext_name,
        sources=op_files,
        include_dirs=[include_path]
    ))

    return extensions


if __name__ == "__main__":
    setup(
        name='ps_vit',
        version='0.0.1',
        description='vision transformer with progressive sampling',
        packages=find_packages(),
        ext_modules=get_extensions(),
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False
    )
