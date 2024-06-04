from setuptools import setup, Extension, find_packages
import os


def main():
    setup(
        name="detection_utils",
        description="Object detection utilities for TensorFlow and YOLO.",
        version="0.1",
        author="Jakub Jach",
        packages=find_packages("src"),
        package_dir={
            '': 'src',
        },
        install_requires=[
            'opencv-python>=4.9',
            'shapely>=2.0.4',
            'pandas',
            'pillow>=9.4',
            'matplotlib>=3.8.4',
            'alive-progress'
            # 'protobuf==3.20.3', # Install this manually
            # 'object-detection', # Clone into models dir, build with protoc and install
            # 'ultralytics', # For YOLO model
            # 'tensorflow==2.10', # Tested on this version
            # + pytorch CUDA/CPU
        ],
        python_requires='>=3.9',
        include_package_data=True
    )


if __name__ == "__main__":
    main()
