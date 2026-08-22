from setuptools import Extension, find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kittentts",
    version="0.8.1",
    author="KittenML",
    author_email="",
    description="Ultra-lightweight text-to-speech model with just 15 million parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kittenml/kittentts",
    packages=find_packages(),
    ext_modules=[
        Extension(
            "kittentts._ephonemizer",
            sources=[
                "kittentts/native_phonemizer/ephonemizer_module.cpp",
                "kittentts/native_phonemizer/phonemizer.cpp",
            ],
            include_dirs=["kittentts/native_phonemizer"],
            language="c++",
            extra_compile_args=["-std=c++17"],
        )
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "espeakng_loader",
        "phonemizer",
        "onnxruntime",
        "soundfile",
        "numpy",
        "huggingface_hub",
    ],
    keywords="text-to-speech, tts, speech-synthesis, neural-networks, onnx",
    project_urls={
        "Bug Reports": "https://github.com/kittenml/kittentts/issues",
        "Source": "https://github.com/kittenml/kittentts",
    },
)
