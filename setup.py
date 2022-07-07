from setuptools import setup, find_packages

setup(
    name="easyvizar-detect",
    version="0.1",
    description="Object detection for EasyVizAR headsets",
    url="https://github.com/EasyVizAR/detect/",

    project_urls = {
        "Homepage": "https://wings.cs.wisc.edu/easyvizar/",
        "Source": "https://github.com/EasyVizAR/detect/",
    },

    packages=find_packages(),

    entry_points={
        "console_scripts": [
            "detect = detect.__main__:main"
        ]
    }
)
