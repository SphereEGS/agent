from setuptools import setup, find_packages

setup(
    name="spherex_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
    ],
    author="EGlobalSphere",
    description="Vehicle tracking and Arabic LPR agent",
    python_requires=">=3.9",
)
