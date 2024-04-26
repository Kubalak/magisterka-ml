from setuptools import setup, Extension, find_packages
import os

def main():
    setup(
        name="detection_utils",
        description="Object detection utilities for TensorFlow and YOLO.",
        version="0.0.1",
        author="Jakub Jach",
        packages=find_packages("src"),
        package_dir={
            '': 'src',
        },

        python_requires='>=3.9',
        include_package_data=True
    )

if __name__ == "__main__":
    main()