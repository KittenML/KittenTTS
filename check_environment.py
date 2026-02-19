#!/usr/bin/env python3
"""
KittenTTS Environment Check Script
Run this to verify your environment is properly configured for best audio quality.
"""

import sys
import subprocess

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"  [OK] {package_name}")
        return True
    except ImportError:
        print(f"  [MISSING] {package_name} (pip install {package_name})")
        return False

def check_version(module_name, attr='__version__'):
    """Get version of a module."""
    try:
        mod = __import__(module_name)
        version = getattr(mod, attr, 'unknown')
        if callable(version):
            version = version()
        return version
    except:
        return 'unknown'

def main():
    print("=" * 60)
    print("KittenTTS Environment Check")
    print("=" * 60)
    
    # Check if in conda environment
    if sys.prefix == sys.base_prefix:
        print("\n[NOTE] Not running in a virtual environment.")
        print("       For best results, use a conda environment:")
        print("       conda activate kittentts")
    else:
        print(f"\n[OK] Running in virtual environment: {sys.prefix}")
    
    all_ok = True
    
    # Check Python version
    print(f"\nPython Version: {sys.version}")
    if sys.version_info < (3, 8):
        print("  [WARN] Python 3.8+ recommended")
        all_ok = False
    else:
        print("  [OK] Python version")
    
    # Check core dependencies
    print("\nCore Dependencies:")
    deps = [
        ('numpy', 'numpy'),
        ('onnxruntime', 'onnxruntime'),
        ('soundfile', 'soundfile'),
        ('phonemizer', 'phonemizer'),
        ('spacy', 'spacy'),
        ('num2words', 'num2words'),
        ('huggingface_hub', 'huggingface-hub'),
    ]
    
    for module, package in deps:
        if not check_import(module, package):
            all_ok = False
    
    # Check specific versions
    print("\nVersions:")
    try:
        import numpy as np
        print(f"  NumPy: {np.__version__}")
        if int(np.__version__.split('.')[0]) >= 2:
            print("    [WARN] NumPy 2.x detected - use numpy<2.0 for best compatibility")
    except:
        pass
        
    try:
        import onnxruntime as ort
        print(f"  ONNX Runtime: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"    Providers: {providers}")
        if 'CPUExecutionProvider' not in providers:
            print("    [WARN] CPUExecutionProvider not available")
    except:
        pass
    
    # Check espeak
    print("\n[Espeak Phonemizer Backend]")
    try:
        # Try to load espeak-ng library if available (needed on Windows)
        try:
            import espeakng_loader
            espeakng_loader.load_library()
            import os
            if 'ESPEAK_DATA_PATH' not in os.environ:
                os.environ['ESPEAK_DATA_PATH'] = str(espeakng_loader.get_data_path())
            # Tell phonemizer where to find the espeak library
            from phonemizer.backend.espeak.base import BaseEspeakBackend
            BaseEspeakBackend.set_library(str(espeakng_loader.get_library_path()))
        except:
            pass
        
        from phonemizer.backend import BACKENDS
        EspeakBackend = BACKENDS.get('espeak') or BACKENDS.get('espeak-ng')
        if EspeakBackend is None:
            raise RuntimeError("No espeak backend available")
        
        # Try to create a backend
        backend = EspeakBackend('en-us')
        print(f"  [OK] Espeak backend working (language: {backend.language})")
        
        # Test phonemization
        test = backend.phonemize(["hello world"])
        # Encoding-safe print (Windows console may not support IPA chars)
        try:
            print(f"  [OK] Phonemization test: {test}")
        except UnicodeEncodeError:
            print(f"  [OK] Phonemization working (output contains IPA characters)")
    except Exception as e:
        print(f"  [ERROR] Espeak backend: {e}")
        print("    Fix: pip install espeakng-loader phonemizer")
        all_ok = False
    
    # Check spacy model
    print("\nSpacy English Model:")
    try:
        import spacy
        try:
            nlp = spacy.load('en_core_web_sm')
            print("  [OK] en_core_web_sm loaded")
        except:
            print("  [MISSING] en_core_web_sm not found")
            print("    Run: python -m spacy download en_core_web_sm")
            all_ok = False
    except:
        pass
    
    # ONNX Runtime optimization check
    print("\nONNX Runtime Configuration:")
    try:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        print(f"  Default threads: {sess_options.intra_op_num_threads}")
        print(f"  Graph optimization: {sess_options.graph_optimization_level}")
        
        # Check if we can load a test session
        print("  [OK] ONNX Runtime session options accessible")
    except Exception as e:
        print(f"  [ERROR] ONNX: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("[SUCCESS] Environment looks good! KittenTTS should work well.")
        print("\nTips for best audio quality:")
        print("   1. Use speed=1.0 for most natural speech")
        print("   2. Keep sentences under 400 characters")
        print("   3. End sentences with punctuation for better prosody")
        print("   4. Use 'Jasper' or 'Bella' for clearest speech")
    else:
        print("[ERROR] Some issues found. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        print("   python -m spacy download en_core_web_sm")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
