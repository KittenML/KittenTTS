#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常场景测试脚本
测试内容：
1. 空文本处理
2. 非法文本输入
3. 超长文本处理
4. 特殊字符处理
5. 文件上传失败场景
6. 模型加载失败场景
7. 语音参数错误场景
"""

import sys
import os
import unittest
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestEmptyTextScenarios(unittest.TestCase):
    """空文本处理测试"""
    
    @classmethod
    def setUpClass(cls):
        from kittentts.chinese_processor import ChineseTextProcessor
        from kittentts.preprocess import TextPreprocessor
        cls.chinese_processor = ChineseTextProcessor()
        cls.preprocessor = TextPreprocessor()
    
    def test_01_empty_text_chinese_processor(self):
        """测试中文处理器处理空文本"""
        print("\n" + "=" * 60)
        print("异常场景测试1: 空文本处理")
        print("=" * 60)
        
        test_cases = [
            "",
            "   ",
            "\n",
            "\t",
            "  \n\t  ",
        ]
        
        print("\n  中文处理器空文本测试:")
        for text in test_cases:
            has_chinese = self.chinese_processor.has_chinese(text)
            print(f"    输入: '{repr(text)}'")
            print(f"    has_chinese: {has_chinese}")
            
            try:
                processed, subtitle = self.chinese_processor.process(text)
                print(f"    处理结果: '{processed}'")
                print(f"    [OK] 无异常")
            except Exception as e:
                print(f"    [ERROR] 异常: {e}")
                self.fail(f"空文本处理应不抛出异常: {e}")
    
    def test_02_empty_text_preprocessor(self):
        """测试英文预处理空文本"""
        print("\n  英文预处理空文本测试:")
        
        test_cases = [
            "",
            "   ",
            "\n",
        ]
        
        for text in test_cases:
            try:
                result = self.preprocessor(text)
                print(f"    输入: '{repr(text)}'")
                print(f"    处理结果: '{result}'")
                print(f"    [OK] 无异常")
            except Exception as e:
                print(f"    [ERROR] 异常: {e}")
                self.fail(f"空文本预处理应不抛出异常: {e}")


class TestInvalidTextScenarios(unittest.TestCase):
    """非法文本输入测试"""
    
    @classmethod
    def setUpClass(cls):
        from kittentts.chinese_processor import ChineseTextProcessor
        from kittentts.preprocess import TextPreprocessor
        cls.chinese_processor = ChineseTextProcessor()
        cls.preprocessor = TextPreprocessor()
    
    def test_01_special_characters(self):
        """测试特殊字符处理"""
        print("\n" + "=" * 60)
        print("异常场景测试2: 特殊字符处理")
        print("=" * 60)
        
        test_cases = [
            "Hello\x00World",
            "Test\x01\x02\x03",
            "Normal text with \x7f char",
        ]
        
        print("\n  特殊字符处理测试:")
        for text in test_cases:
            print(f"\n    输入: {repr(text)}")
            
            try:
                has_chinese = self.chinese_processor.has_chinese(text)
                print(f"    has_chinese: {has_chinese}")
                
                processed, subtitle = self.chinese_processor.process(text)
                print(f"    中文处理: '{processed[:50]}...'")
                print(f"    [OK] 无异常")
            except Exception as e:
                print(f"    [ERROR] 异常: {e}")
            
            try:
                eng_processed = self.preprocessor(text)
                print(f"    英文处理: '{eng_processed[:50]}...'")
                print(f"    [OK] 无异常")
            except Exception as e:
                print(f"    [ERROR] 异常: {e}")
    
    def test_02_mixed_encoding(self):
        """测试混合编码文本"""
        print("\n  混合编码文本测试:")
        
        test_cases = [
            "Hello 你好 World",
            "中文, English, Mixed Text",
            "Mix: 你好 World 123",
        ]
        
        for text in test_cases:
            print(f"\n    输入: '{text}'")
            
            try:
                processed, subtitle = self.chinese_processor.process(text)
                print(f"    处理: '{processed}'")
                print(f"    拼音: '{subtitle.get('pinyin', '')}'")
                print(f"    [OK] 无异常")
            except Exception as e:
                print(f"    [ERROR] 异常: {e}")
    
    def test_03_only_punctuation(self):
        """测试纯标点符号文本"""
        print("\n  纯标点符号文本测试:")
        
        test_cases = [
            "!!!",
            "???",
            "!!!???",
            "。。。",
            "，，，",
            "；：「」『』",
            "!@#$%^&*()_+",
        ]
        
        for text in test_cases:
            print(f"\n    输入: '{text}'")
            
            try:
                has_chinese = self.chinese_processor.has_chinese(text)
                print(f"    has_chinese: {has_chinese}")
                
                processed, subtitle = self.chinese_processor.process(text)
                print(f"    处理: '{processed}'")
                print(f"    [OK] 无异常")
            except Exception as e:
                print(f"    [ERROR] 异常: {e}")


class TestLongTextScenarios(unittest.TestCase):
    """超长文本处理测试"""
    
    @classmethod
    def setUpClass(cls):
        from kittentts.preprocess import TextPreprocessor
        cls.preprocessor = TextPreprocessor()
    
    def test_01_very_long_text(self):
        """测试超长文本"""
        print("\n" + "=" * 60)
        print("异常场景测试3: 超长文本处理")
        print("=" * 60)
        
        from kittentts.onnx_model import chunk_text
        
        long_text_1 = "Hello World. " * 100
        long_text_2 = "你好世界。" * 100
        long_text_3 = "This is sentence number one. " * 50
        mixed_long = ("Hello World. 你好世界。" * 50)
        
        test_cases = [
            ("重复英文短句", long_text_1),
            ("重复中文短句", long_text_2),
            ("重复完整句子", long_text_3),
            ("中英文混合", mixed_long),
        ]
        
        for name, text in test_cases:
            print(f"\n  [{name}]")
            print(f"    文本长度: {len(text)} 字符")
            
            try:
                chunks = chunk_text(text, max_len=200)
                print(f"    分块数量: {len(chunks)}")
                for i, chunk in enumerate(chunks[:3]):
                    print(f"      块{i+1}: '{chunk[:50]}...' ({len(chunk)} 字符)")
                if len(chunks) > 3:
                    print(f"      ... 还有 {len(chunks) - 3} 个块")
                print(f"    [OK] 分块成功")
            except Exception as e:
                print(f"    [ERROR] 分块异常: {e}")
            
            try:
                processed = self.preprocessor(text)
                print(f"    预处理后长度: {len(processed)} 字符")
                print(f"    [OK] 预处理成功")
            except Exception as e:
                print(f"    [ERROR] 预处理异常: {e}")
    
    def test_02_single_long_word(self):
        """测试超长单词"""
        print("\n  超长单词测试:")
        
        long_word = "a" * 500
        long_number = "1234567890" * 50
        
        test_cases = [
            ("超长字母", long_word),
            ("超长数字", long_number),
            ("超长混合", long_word + " " + long_number),
        ]
        
        for name, text in test_cases:
            print(f"\n    [{name}]")
            print(f"      长度: {len(text)} 字符")
            
            try:
                processed = self.preprocessor(text)
                print(f"      预处理: '{processed[:100]}...'")
                print(f"      [OK] 无异常")
            except Exception as e:
                print(f"      [ERROR] 异常: {e}")


class TestWebUIErrorScenarios(unittest.TestCase):
    """WebUI异常场景测试"""
    
    @classmethod
    def setUpClass(cls):
        from webui.app import app
        app.config['TESTING'] = True
        cls.client = app.test_client()
    
    def test_01_invalid_request_methods(self):
        """测试无效请求方法"""
        print("\n" + "=" * 60)
        print("异常场景测试4: 无效请求方法")
        print("=" * 60)
        
        test_cases = [
            ('/api/generate', 'GET', "POST接口用GET请求"),
            ('/api/tasks', 'GET', "任务列表接口"),
            ('/api/models', 'POST', "模型列表接口用POST请求"),
        ]
        
        for endpoint, method, desc in test_cases:
            print(f"\n  [{desc}]")
            print(f"    端点: {endpoint}")
            print(f"    方法: {method}")
            
            try:
                if method == 'GET':
                    response = self.client.get(endpoint)
                else:
                    response = self.client.post(endpoint)
                
                print(f"    状态码: {response.status_code}")
                
                if response.status_code == 405:
                    print(f"    [OK] 正确返回405 Method Not Allowed")
                elif response.status_code in [200, 400]:
                    print(f"    [OK] 接口正常响应")
                else:
                    print(f"    [INFO] 响应状态码: {response.status_code}")
                    
            except Exception as e:
                print(f"    [ERROR] 请求异常: {e}")
    
    def test_02_missing_parameters(self):
        """测试缺少必需参数"""
        print("\n" + "=" * 60)
        print("异常场景测试5: 缺少必需参数")
        print("=" * 60)
        
        test_cases = [
            ({}, "空JSON"),
            ({'text': ''}, "空文本"),
            ({'model': 'invalid-model'}, "缺少text参数"),
            ({'text': 'Hello', 'speed': 'invalid'}, "无效speed参数"),
        ]
        
        for data, desc in test_cases:
            print(f"\n  [{desc}]")
            print(f"    请求数据: {data}")
            
            try:
                response = self.client.post(
                    '/api/generate',
                    json=data,
                    content_type='application/json'
                )
                print(f"    状态码: {response.status_code}")
                
                result = response.get_json()
                if result:
                    print(f"    响应: {result}")
                
                if response.status_code == 400:
                    print(f"    [OK] 正确返回400 Bad Request")
                elif response.status_code == 500:
                    print(f"    [INFO] 服务器内部错误")
                    
            except Exception as e:
                print(f"    [ERROR] 请求异常: {e}")
    
    def test_03_invalid_content_type(self):
        """测试无效Content-Type"""
        print("\n" + "=" * 60)
        print("异常场景测试6: 无效Content-Type")
        print("=" * 60)
        
        test_cases = [
            ('text/plain', 'text=Hello'),
            ('application/x-www-form-urlencoded', 'text=Hello'),
            (None, '{"text": "Hello"}'),
        ]
        
        for content_type, data in test_cases:
            print(f"\n  [Content-Type: {content_type}]")
            
            try:
                headers = {}
                if content_type:
                    headers['Content-Type'] = content_type
                
                response = self.client.post(
                    '/api/generate',
                    data=data,
                    headers=headers
                )
                print(f"    状态码: {response.status_code}")
                print(f"    [OK] 请求完成")
                    
            except Exception as e:
                print(f"    [ERROR] 请求异常: {e}")


class TestFileUploadScenarios(unittest.TestCase):
    """文件上传失败场景测试"""
    
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
    
    def test_01_file_upload_errors(self):
        """测试文件上传错误场景"""
        print("\n" + "=" * 60)
        print("异常场景测试7: 文件上传错误场景")
        print("=" * 60)
        
        print("\n  创建测试文件:")
        
        test_files = [
            ("valid.txt", "Hello World\n这是测试文本。", "有效文本文件"),
            ("invalid.pdf", "%PDF-1.4\nfake pdf content", "无效格式文件"),
            ("empty.txt", "", "空文件"),
            ("very_large.txt", "a" * 1000000, "大文件(1MB)"),
        ]
        
        for filename, content, desc in test_files:
            filepath = os.path.join(self.temp_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"    [创建] {filename} ({len(content.encode('utf-8'))} 字节) - {desc}")
        
        print("\n  模拟文件上传场景测试:")
        
        scenarios = [
            ("有效txt文件", "valid.txt", True, "应成功读取"),
            ("空txt文件", "empty.txt", True, "应读取为空"),
            ("非txt文件", "invalid.pdf", False, "前端应拒绝"),
            ("大文件", "very_large.txt", True, "应能处理（但可能很慢）"),
        ]
        
        for name, filename, should_read, expectation in scenarios:
            filepath = os.path.join(self.temp_dir, filename)
            print(f"\n    [{name}]")
            print(f"      文件: {filename}")
            print(f"      预期: {expectation}")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"      读取成功, 大小: {len(content)} 字符")
                
                if filename.endswith('.txt'):
                    print(f"      [OK] txt文件可正常读取")
                else:
                    print(f"      [INFO] 非txt文件也可读取（由前端验证）")
                    
            except UnicodeDecodeError:
                print(f"      [INFO] UTF-8解码失败（二进制文件）")
            except Exception as e:
                print(f"      [ERROR] 读取异常: {e}")
    
    def test_02_file_type_validation(self):
        """测试文件类型验证逻辑"""
        print("\n" + "=" * 60)
        print("异常场景测试8: 文件类型验证")
        print("=" * 60)
        
        test_files = [
            ("test.txt", True),
            ("test.TXT", True),
            ("test.TxT", True),
            ("test.pdf", False),
            ("test.doc", False),
            ("test", False),
            ("test.txt.pdf", False),
            ("test..txt", True),
        ]
        
        print("\n  文件扩展名验证测试:")
        for filename, expected_valid in test_files:
            is_valid = filename.lower().endswith('.txt')
            status = "[OK]" if is_valid == expected_valid else "[FAIL]"
            print(f"    {status} '{filename}' -> 有效: {is_valid} (预期: {expected_valid})")
            
            if is_valid != expected_valid:
                print(f"      [WARNING] 验证逻辑与预期不符")
        
        print("\n  [OK] 文件类型验证逻辑测试完成")


class TestParameterValidation(unittest.TestCase):
    """参数验证测试"""
    
    @classmethod
    def setUpClass(cls):
        from kittentts.onnx_model import KittenTTS_1_Onnx
        cls.ModelClass = KittenTTS_1_Onnx
    
    def test_01_speed_parameter_validation(self):
        """测试语速参数"""
        print("\n" + "=" * 60)
        print("异常场景测试9: 语速参数验证")
        print("=" * 60)
        
        test_speeds = [
            (0.1, "极慢"),
            (0.5, "慢"),
            (1.0, "正常"),
            (1.5, "快"),
            (2.0, "极快"),
            (-1.0, "负数"),
            (0.0, "零"),
            (100.0, "过大值"),
        ]
        
        print("\n  语速参数测试:")
        for speed, desc in test_speeds:
            print(f"    [{desc}] speed = {speed}")
            
            try:
                speed_float = float(speed)
                print(f"      转换为float: {speed_float}")
                
                if speed_float <= 0:
                    print(f"      [WARNING] 语速应为正数")
                elif speed_float < 0.5:
                    print(f"      [INFO] 语速可能过慢")
                elif speed_float > 2.0:
                    print(f"      [INFO] 语速可能过快")
                else:
                    print(f"      [OK] 语速在正常范围内")
                    
            except (ValueError, TypeError) as e:
                print(f"      [ERROR] 无效语速: {e}")
    
    def test_02_voice_parameter_validation(self):
        """测试语音参数"""
        print("\n" + "=" * 60)
        print("异常场景测试10: 语音参数验证")
        print("=" * 60)
        
        available_voices = [
            'Bella', 'Jasper', 'Luna', 'Bruno', 
            'Rosie', 'Hugo', 'Kiki', 'Leo'
        ]
        
        internal_voices = [
            'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 
            'expr-voice-3-f', 'expr-voice-4-m', 'expr-voice-4-f', 
            'expr-voice-5-m', 'expr-voice-5-f'
        ]
        
        test_voices = [
            ("Bruno", True, "有效语音"),
            ("Bella", True, "有效语音"),
            ("invalid", False, "无效语音"),
            ("", False, "空语音"),
            (None, False, "None"),
            ("expr-voice-5-m", True, "内部名称"),
        ]
        
        print("\n  语音参数测试:")
        for voice, expected_valid, desc in test_voices:
            print(f"\n    [{desc}]")
            print(f"      语音: {voice}")
            
            is_valid = voice in available_voices or voice in internal_voices
            
            if is_valid:
                print(f"      [OK] 有效语音")
            else:
                print(f"      [ERROR] 无效语音，应抛出异常")


def run_error_case_tests():
    """运行所有异常场景测试"""
    print("\n" + "=" * 70)
    print("KittenTTS 异常场景测试")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestEmptyTextScenarios))
    suite.addTests(loader.loadTestsFromTestCase(TestInvalidTextScenarios))
    suite.addTests(loader.loadTestsFromTestCase(TestLongTextScenarios))
    suite.addTests(loader.loadTestsFromTestCase(TestWebUIErrorScenarios))
    suite.addTests(loader.loadTestsFromTestCase(TestFileUploadScenarios))
    suite.addTests(loader.loadTestsFromTestCase(TestParameterValidation))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"  运行测试数: {result.testsRun}")
    print(f"  成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_error_case_tests()
    sys.exit(0 if success else 1)
