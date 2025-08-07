#!/usr/bin/env python3
"""
Setup script for Enhanced KittenTTS - Real-time Text-to-Speech Synthesis
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Read version from __init__.py
def get_version():
    init_file = Path(__file__).parent / "kittentts" / "__init__.py"
    if init_file.exists():
        with open(init_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return "0.2.0"

# Read long description from README
def get_long_description():
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return """
Enhanced KittenTTS: Ultra-lightweight real-time text-to-speech model with advanced streaming capabilities.

Features:
- Real-time synthesis with sub-second latency
- Streaming audio generation for interactive applications
- Advanced caching and performance optimizations
- Parallel processing with configurable workers
- Adaptive quality management
- Comprehensive monitoring and profiling
- GPU acceleration support
"""

# Core requirements for basic functionality
CORE_REQUIREMENTS = [
    "numpy>=1.19.0",
    "onnxruntime>=1.12.0", 
    "huggingface-hub>=0.10.0",
    "soundfile>=0.10.0",
    "phonemizer>=3.0.0",
    "misaki>=0.1.0",
]

# Optional requirements for enhanced features
OPTIONAL_REQUIREMENTS = {
    # Real-time and streaming features
    "realtime": [
        "threading-extensions>=0.1.0",
    ],
    
    # Performance monitoring
    "monitoring": [
        "psutil>=5.8.0",
        "memory-profiler>=0.60.0",
    ],
    
    # GPU acceleration
    "gpu": [
        "onnxruntime-gpu>=1.12.0",
    ],
    
    # Advanced audio processing
    "audio": [
        "librosa>=0.8.0",
        "scipy>=1.7.0",
        "resampy>=0.2.0",
    ],
    
    # Development and testing
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
        "black>=21.0.0",
        "flake8>=3.9.0",
        "mypy>=0.910",
        "pre-commit>=2.15.0",
    ],
    
    # Documentation
    "docs": [
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=0.5.0",
        "myst-parser>=0.15.0",
    ],
    
    # Examples and demos
    "examples": [
        "jupyter>=1.0.0",
        "matplotlib>=3.3.0",
        "ipywidgets>=7.6.0",
    ]
}

# All optional requirements combined
OPTIONAL_REQUIREMENTS["all"] = [
    req for req_list in OPTIONAL_REQUIREMENTS.values() 
    for req in req_list if req_list != OPTIONAL_REQUIREMENTS["all"]
]

# Platform-specific requirements
if sys.platform.startswith('win'):
    CORE_REQUIREMENTS.extend([
        "pywin32>=227",
    ])
elif sys.platform.startswith('darwin'):
    CORE_REQUIREMENTS.extend([
        "pyobjc-framework-AVFoundation>=7.0",
    ])

# Check Python version
if sys.version_info < (3, 7):
    print("Error: KittenTTS requires Python 3.7 or higher.")
    sys.exit(1)

# Custom commands
class DevelopCommand(setuptools.Command):
    """Custom command for development setup"""
    description = "Install package in development mode with all optional dependencies"
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        os.system("pip install -e .[all,dev]")
        os.system("pre-commit install")
        print("Development environment setup complete!")

# Entry points for command-line tools
ENTRY_POINTS = {
    'console_scripts': [
        'kittentts=kittentts.cli:main',
        'kittentts-demo=kittentts.examples.demo:main',
        'kittentts-benchmark=kittentts.examples.benchmark:main',
        'kittentts-server=kittentts.server:main',
    ],
}

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8", 
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Multimedia :: Sound/Audio :: Speech",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: System :: Hardware :: Universal Serial Bus (USB) :: Audio (UDA)",
]

# Project URLs
PROJECT_URLS = {
    "Homepage": "https://github.com/KittenML/kitten-tts",
    "Bug Reports": "https://github.com/KittenML/kitten-tts/issues", 
    "Source": "https://github.com/KittenML/kitten-tts",
    "Documentation": "https://kitten-tts.readthedocs.io/",
    "Changelog": "https://github.com/KittenML/kitten-tts/blob/main/CHANGELOG.md",
}

# Package data and resources
PACKAGE_DATA = {
    "kittentts": [
        "configs/*.json",
        "models/*.onnx",
        "voices/*.npz",
        "examples/*.py",
        "examples/audio/*.wav",
        "templates/*.html",
        "static/*.css",
        "static/*.js",
    ]
}

# Setup configuration
setup(
    name="kittentts",
    version=get_version(),
    author="Humair_Munir",
    author_email="humairmunirawan@gmail.com",
    description="Ultra-lightweight real-time text-to-speech model with advanced streaming capabilities",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/KittenML/kitten-tts",
    project_urls=PROJECT_URLS,
    
    # Package configuration
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    package_data=PACKAGE_DATA,
    include_package_data=True,
    zip_safe=False,
    
    # Requirements
    python_requires=">=3.7",
    install_requires=CORE_REQUIREMENTS,
    extras_require=OPTIONAL_REQUIREMENTS,
    
    # Metadata
    classifiers=CLASSIFIERS,
    keywords=[
        "text-to-speech", "tts", "speech-synthesis", "neural-tts", 
        "real-time", "streaming", "voice", "audio", "ai", "ml",
        "onnx", "lightweight", "fast", "kitten"
    ],
    license="Apache License 2.0",
    
    # Entry points
    entry_points=ENTRY_POINTS,
    
    # Custom commands
    cmdclass={
        'develop': DevelopCommand,
    },
    
    # Additional options
    options={
        'bdist_wheel': {
            'universal': False,  # Pure Python but version-specific
        },
        'egg_info': {
            'tag_build': None,
            'tag_date': False,
        },
    },
)

# Post-installation message
def post_install_message():
    """Display post-installation instructions"""
    message = """
╭─────────────────────────────────────────────────────────────╮
│                  KittenTTS Installation Complete!           │
├─────────────────────────────────────────────────────────────┤
│                                                            │
│ 🚀 Quick Start:                                           │
│   >>> from kittentts import KittenTTS                     │
│   >>> tts = KittenTTS(realtime_mode=True)                 │
│   >>> audio = tts.generate("Hello, world!")               │
│                                                            │
│ 🎯 Real-time Features:                                    │
│   • Sub-second generation latency                          │
│   • Streaming synthesis capabilities                       │
│   • Parallel processing (up to 8x speedup)                │
│   • Advanced caching (10-100x speedups)                   │
│   • Adaptive quality management                            │
│                                                            │
│ 🛠  Command Line Tools:                                   │
│   kittentts --help          # Main CLI interface          │
│   kittentts-demo            # Run feature demonstrations   │
│   kittentts-benchmark       # Performance benchmarking     │
│   kittentts-server          # Start TTS server            │
│                                                            │
│ 📖 Documentation: https://kitten-tts.readthedocs.io/     │
│ 🐛 Issues: https://github.com/KittenML/kitten-tts/issues │
│                                                            │
╰─────────────────────────────────────────────────────────────╯
    """
    print(message)

# Display post-install message if this is being run as main
if __name__ == "__main__" and "install" in sys.argv:
    import atexit
    atexit.register(post_install_message)

# Additional setup validation
def validate_installation():
    """Validate that the installation will work correctly"""
    try:
        import numpy
        import onnxruntime
        print("✓ Core dependencies validated")
    except ImportError as e:
        print(f"❌ Core dependency missing: {e}")
        print("Please ensure all required packages are installed")
        return False
    
    # Check ONNX runtime providers
    try:
        providers = onnxruntime.get_available_providers()
        print(f"✓ ONNX Runtime providers: {', '.join(providers)}")
        if 'CUDAExecutionProvider' in providers:
            print("✓ GPU acceleration available")
        else:
            print("ℹ  GPU acceleration not available (CPU only)")
    except Exception as e:
        print(f"⚠  Could not check ONNX providers: {e}")
    
    return True

# Run validation if requested
if "--validate" in sys.argv:
    sys.argv.remove("--validate")
    if not validate_installation():
        sys.exit(1)

# Development helper functions
def create_dev_scripts():
    """Create development helper scripts"""
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    # Test script
    test_script = scripts_dir / "test.py"
    test_script.write_text("""#!/usr/bin/env python3
import subprocess
import sys

def run_tests():
    \"\"\"Run test suite\"\"\"
    try:
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--cov=kittentts"], check=True)
    except subprocess.CalledProcessError:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
""")
    
    # Benchmark script
    bench_script = scripts_dir / "benchmark.py"
    bench_script.write_text("""#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")

from kittentts.examples.example_usage import run_comprehensive_benchmark

if __name__ == "__main__":
    run_comprehensive_benchmark()
""")
    
    # Make scripts executable
    import stat
    for script in [test_script, bench_script]:
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
    
    print(f"✓ Development scripts created in {scripts_dir}/")

# Create development scripts if in development mode
if "--create-dev-scripts" in sys.argv:
    sys.argv.remove("--create-dev-scripts")
    create_dev_scripts()

print("Setup configuration complete!")
