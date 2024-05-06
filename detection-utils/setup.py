from setuptools import setup, Extension, find_packages
import os

def main():
    setup(
        name="detection_utils",
        description="Object detection utilities for TensorFlow and YOLO.",
        version="0.0.2",
        author="Jakub Jach",
        packages=find_packages("src"),
        package_dir={
            '': 'src',
        },
        install_requires=[
            'opencv-python>=4.9',
            'shapely>=2.0.4',
            'matplotlib>=3.8.4',
            'tensorflow==2.10.0',
            'alive-progress',
            'protobuf==3.20.3'
        ],
        python_requires='>=3.9',
        include_package_data=True
    )

if __name__ == "__main__":
    main()