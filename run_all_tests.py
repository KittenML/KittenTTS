#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KittenTTS 综合测试入口脚本
运行所有测试套件并生成测试报告
"""

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """检查项目依赖"""
    print("\n" + "=" * 70)
    print("检查项目依赖")
    print("=" * 70)
    
    required_packages = [
        'numpy',
        'onnxruntime',
        'soundfile',
        'phonemizer',
        'huggingface_hub',
    ]
    
    optional_packages = [
        'pypinyin',
        'cn2an',
        'flask',
        'werkzeug',
    ]
    
    print("\n必需依赖:")
    all_required = True
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [FAIL] {pkg} - 未安装")
            all_required = False
    
    print("\n可选依赖（中文支持和WebUI）:")
    for pkg in optional_packages:
        try:
            __import__(pkg)
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [!] {pkg} - 未安装（可选）")
    
    print("\n" + "-" * 70)
    if all_required:
        print("[OK] 所有必需依赖已安装")
    else:
        print("[FAIL] 缺少必需依赖，请运行: pip install -r requirements.txt")
    
    return all_required


def check_project_structure():
    """检查项目结构"""
    print("\n" + "=" * 70)
    print("检查项目结构")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_files = [
        ('kittentts/__init__.py', "核心模块初始化"),
        ('kittentts/preprocess.py', "英文文本预处理"),
        ('kittentts/chinese_processor.py', "中文处理器"),
        ('kittentts/onnx_model.py', "ONNX模型推理"),
        ('kittentts/get_model.py', "模型加载"),
        ('webui/app.py', "WebUI后端"),
        ('webui/templates/index.html', "WebUI前端"),
        ('requirements.txt', "依赖文件"),
    ]
    
    print("\n必需文件:")
    all_found = True
    for filepath, desc in required_files:
        full_path = os.path.join(base_dir, filepath)
        if os.path.exists(full_path):
            print(f"  [OK] {filepath} - {desc}")
        else:
            print(f"  [FAIL] {filepath} - 缺失！")
            all_found = False
    
    test_files = [
        ('test_backend.py', "后端单元测试"),
        ('test_webui_api.py', "WebUI API测试"),
        ('test_error_cases.py', "异常场景测试"),
    ]
    
    print("\n测试文件:")
    for filepath, desc in test_files:
        full_path = os.path.join(base_dir, filepath)
        if os.path.exists(full_path):
            print(f"  [OK] {filepath} - {desc}")
        else:
            print(f"  [!] {filepath} - 未找到")
    
    return all_found


def run_test_suite(suite_name, test_loader, test_classes):
    """运行测试套件"""
    print("\n" + "=" * 70)
    print(f"运行测试套件: {suite_name}")
    print("=" * 70)
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        suite.addTests(test_loader.loadTestsFromTestCase(test_class))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


def generate_report(results, start_time, end_time):
    """生成测试报告"""
    print("\n" + "=" * 70)
    print("测试报告")
    print("=" * 70)
    
    duration = end_time - start_time
    
    total_tests = sum(r.testsRun for r in results)
    total_failures = sum(len(r.failures) for r in results)
    total_errors = sum(len(r.errors) for r in results)
    total_passed = total_tests - total_failures - total_errors
    
    print(f"\n执行时间: {duration:.2f} 秒")
    print(f"\n测试统计:")
    print(f"  总测试数: {total_tests}")
    print(f"  用例通过: {total_passed}")
    print(f"  用例失败: {total_failures}")
    print(f"  用例错误: {total_errors}")
    
    if total_failures == 0 and total_errors == 0:
        print(f"\n[OK] 所有测试通过！")
        status = "PASS"
    else:
        print(f"\n[FAIL] 部分测试失败，请检查详细输出")
        status = "FAIL"
    
    print("\n" + "=" * 70)
    
    return status == "PASS"


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    KittenTTS 综合测试套件                     ║
║                    =======================                     ║
║  测试内容:                                                    ║
║  1. 后端推理服务（中文/英文文本处理、模型加载）               ║
║  2. WebUI API（模型列表、语音列表、音频生成、任务队列）      ║
║  3. 异常场景（空文本、非法输入、文件上传失败）               ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    print("\n[步骤 1] 检查环境")
    deps_ok = check_dependencies()
    struct_ok = check_project_structure()
    
    if not deps_ok or not struct_ok:
        print("\n[!] 环境检查未通过，是否继续运行测试？")
        response = input("  输入 'y' 继续，其他键退出: ").strip().lower()
        if response != 'y':
            print("\n测试已取消。")
            sys.exit(1)
    
    print("\n[步骤 2] 加载测试模块")
    
    try:
        from test_backend import (
            TestChineseProcessor,
            TestEnglishPreprocessor,
            TestTextCleaner,
            TestModelLoading
        )
        print("  [OK] 加载后端测试模块")
    except Exception as e:
        print(f"  [FAIL] 加载后端测试模块失败: {e}")
        return False
    
    try:
        from test_webui_api import (
            TestWebUIAPIBasic,
            TestWebUIAPIErrorCases,
            TestWebUIAPIFullFlow
        )
        print("  [OK] 加载WebUI API测试模块")
    except Exception as e:
        print(f"  [FAIL] 加载WebUI API测试模块失败: {e}")
        return False
    
    try:
        from test_error_cases import (
            TestEmptyTextScenarios,
            TestInvalidTextScenarios,
            TestLongTextScenarios,
            TestWebUIErrorScenarios,
            TestFileUploadScenarios,
            TestParameterValidation
        )
        print("  [OK] 加载异常场景测试模块")
    except Exception as e:
        print(f"  [FAIL] 加载异常场景测试模块失败: {e}")
        return False
    
    loader = unittest.TestLoader()
    results = []
    
    start_time = time.time()
    
    print("\n[步骤 3] 运行后端单元测试")
    backend_classes = [
        TestChineseProcessor,
        TestEnglishPreprocessor,
        TestTextCleaner,
    ]
    result = run_test_suite("后端单元测试", loader, backend_classes)
    results.append(result)
    
    print("\n[步骤 4] 运行WebUI API测试")
    webui_classes = [
        TestWebUIAPIBasic,
        TestWebUIAPIErrorCases,
        TestWebUIAPIFullFlow,
    ]
    result = run_test_suite("WebUI API测试", loader, webui_classes)
    results.append(result)
    
    print("\n[步骤 5] 运行异常场景测试")
    error_classes = [
        TestEmptyTextScenarios,
        TestInvalidTextScenarios,
        TestLongTextScenarios,
        TestWebUIErrorScenarios,
        TestFileUploadScenarios,
        TestParameterValidation,
    ]
    result = run_test_suite("异常场景测试", loader, error_classes)
    results.append(result)
    
    end_time = time.time()
    
    print("\n[步骤 6] 生成测试报告")
    all_passed = generate_report(results, start_time, end_time)
    
    if all_passed:
        print("""
╔══════════════════════════════════════════════════════════════╗
║                      所有测试通过！                           ║
║                                                              ║
║  项目状态: 可部署                                            ║
║  建议操作: 运行 python run_webui.py 启动WebUI              ║
╚══════════════════════════════════════════════════════════════╝
""")
    else:
        print("""
╔══════════════════════════════════════════════════════════════╗
║                    部分测试未通过                            ║
║                                                              ║
║  请检查:                                                     ║
║  1. 依赖是否完整安装                                         ║
║  2. 项目结构是否完整                                         ║
║  3. 查看详细错误信息                                         ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断。")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
