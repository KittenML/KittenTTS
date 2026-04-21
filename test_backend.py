#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后端推理服务单元测试脚本
测试内容：
1. 中文处理器功能测试
2. 英文文本预处理测试
3. 文本解析和音素转换测试
4. 模型加载和语音生成测试（可选，需要网络下载模型）
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestChineseProcessor(unittest.TestCase):
    """中文处理器测试"""
    
    @classmethod
    def setUpClass(cls):
        from kittentts.chinese_processor import ChineseTextProcessor, check_dependencies
        cls.processor = ChineseTextProcessor()
        cls.deps = check_dependencies()
    
    def test_01_dependencies_check(self):
        """测试依赖检查"""
        print("\n" + "=" * 60)
        print("测试1: 依赖检查")
        print("=" * 60)
        print(f"  pypinyin: {'[OK] 已安装' if self.deps['pypinyin'] else '[FAIL] 未安装'}")
        print(f"  cn2an: {'[OK] 已安装' if self.deps['cn2an'] else '[FAIL] 未安装'}")
        print(f"  完整支持: {'[OK]' if self.deps['full_support'] else '[FAIL]'}")
    
    def test_02_has_chinese(self):
        """测试中文字符检测"""
        print("\n" + "=" * 60)
        print("测试2: 中文字符检测")
        print("=" * 60)
        
        test_cases = [
            ("你好世界", True, "纯中文"),
            ("Hello World", False, "纯英文"),
            ("你好 World", True, "中英文混合"),
            ("12345", False, "纯数字"),
            ("你好123", True, "中文加数字"),
            ("", False, "空字符串"),
        ]
        
        all_passed = True
        for text, expected, desc in test_cases:
            result = self.processor.has_chinese(text)
            status = "[OK]" if result == expected else "[FAIL]"
            if result != expected:
                all_passed = False
            print(f"  {status} [{desc}]")
            print(f"    输入: '{text}'")
            print(f"    期望: {expected}, 实际: {result}")
        
        self.assertTrue(all_passed)
    
    def test_03_normalize_punctuation(self):
        """测试中文标点符号规范化"""
        print("\n" + "=" * 60)
        print("测试3: 中文标点符号规范化")
        print("=" * 60)
        
        test_cases = [
            ("你好，世界！", "你好, 世界!", "逗号和感叹号"),
            ("这是【测试】文本。", "这是[测试]文本.", "方括号"),
            ("价格是：100元", "价格是: 100元", "冒号"),
            ("你好吗？", "你好吗?", "问号"),
            ("他说：「你好」", "他说: \"你好\"", "引号"),
        ]
        
        for text, expected, desc in test_cases:
            result = self.processor.normalize_punctuation(text)
            print(f"  [{desc}]")
            print(f"    输入: '{text}'")
            print(f"    输出: '{result}'")
    
    def test_04_number_to_chinese(self):
        """测试阿拉伯数字转中文"""
        print("\n" + "=" * 60)
        print("测试4: 阿拉伯数字转中文")
        print("=" * 60)
        
        from kittentts.chinese_processor import number_to_chinese_words, float_to_chinese_words
        
        test_cases_int = [
            (0, "零"),
            (1, "一"),
            (10, "十"),
            (11, "十一"),
            (123, "一百二十三"),
            (10000, "一万"),
            (1001, "一千零一"),
            (-5, "负五"),
        ]
        
        test_cases_float = [
            ("3.14", "三点一四"),
            ("0.5", "零点五"),
            ("-0.25", "负零点二五"),
            ("100.00", "一百点零零"),
        ]
        
        print("  整数测试:")
        for num, expected in test_cases_int:
            result = number_to_chinese_words(num)
            print(f"    {num} -> {result}")
        
        print("  浮点数测试:")
        for num, expected in test_cases_float:
            result = float_to_chinese_words(num)
            print(f"    {num} -> {result}")
    
    def test_05_chinese_number_conversion(self):
        """测试中文语境下的数字转换"""
        print("\n" + "=" * 60)
        print("测试5: 中文语境下的数字转换")
        print("=" * 60)
        
        from kittentts.chinese_processor import chinese_number_to_words
        
        test_cases = [
            "我有123个苹果。",
            "价格是3.14元。",
            "今天是2024年。",
            "他的身高是1.75米。",
        ]
        
        for text in test_cases:
            result = chinese_number_to_words(text)
            print(f"  输入: '{text}'")
            print(f"  输出: '{result}'")
            print()
    
    @unittest.skipUnless(True, "拼音转换测试")
    def test_06_chinese_to_pinyin(self):
        """测试中文转拼音"""
        print("\n" + "=" * 60)
        print("测试6: 中文转拼音")
        print("=" * 60)
        
        if not self.deps['pypinyin']:
            print("  [!] pypinyin 未安装，跳过拼音测试")
            return
        
        test_cases = [
            "你好世界",
            "我是中国人",
            "今天天气真好",
            "Hello你好World",
        ]
        
        for text in test_cases:
            pinyin_no_tone = self.processor.to_pinyin(text, tone=False)
            pinyin_with_tone = self.processor.to_pinyin(text, tone=True)
            print(f"  输入: '{text}'")
            print(f"  拼音(无声调): '{pinyin_no_tone}'")
            print(f"  拼音(有声调): '{pinyin_with_tone}'")
            print()
    
    def test_07_split_mixed_text(self):
        """测试中英文混合文本分割"""
        print("\n" + "=" * 60)
        print("测试7: 中英文混合文本分割")
        print("=" * 60)
        
        from kittentts.chinese_processor import split_mixed_text
        
        test_cases = [
            "你好Hello世界",
            "I love 你中国",
            "这是一个mixed test：中文和English混合。",
            "纯中文文本",
            "Pure English text",
            "123数字456",
        ]
        
        for text in test_cases:
            segments = split_mixed_text(text)
            print(f"  输入: '{text}'")
            print(f"  分段:")
            for content, seg_type in segments:
                print(f"    [{seg_type}] -> '{content}'")
            print()
    
    def test_08_process_mixed_text(self):
        """测试混合文本处理（用于TTS合成）"""
        print("\n" + "=" * 60)
        print("测试8: 混合文本处理（TTS合成）")
        print("=" * 60)
        
        from kittentts.chinese_processor import process_mixed_text
        
        test_cases = [
            "你好，世界！",
            "Hello, World!",
            "我是中国人，I love China。",
            "价格是3.14元，very expensive。",
        ]
        
        for text in test_cases:
            processed, original = process_mixed_text(text, for_synthesis=True)
            print(f"  原文: '{original}'")
            print(f"  处理: '{processed}'")
            print()
    
    def test_09_extract_subtitle_info(self):
        """测试字幕信息提取"""
        print("\n" + "=" * 60)
        print("测试9: 字幕信息提取")
        print("=" * 60)
        
        from kittentts.chinese_processor import extract_subtitle_info
        
        test_cases = [
            "你好世界",
            "Hello World",
            "我是中国人，I love China。",
        ]
        
        for text in test_cases:
            subtitle = extract_subtitle_info(text)
            print(f"  原文: '{text}'")
            print(f"  原文: {subtitle.get('original', '')}")
            print(f"  拼音: {subtitle.get('pinyin', '')}")
            print(f"  分段数: {len(subtitle.get('segments', []))}")
            print()
    
    def test_10_full_process(self):
        """测试完整文本处理流程"""
        print("\n" + "=" * 60)
        print("测试10: 完整文本处理流程")
        print("=" * 60)
        
        test_cases = [
            "你好，我是KittenTTS！",
            "今天天气真好，气温是25度。",
            "价格是3.14元，非常便宜。",
            "Hello，我是一个混合文本Test。",
        ]
        
        for text in test_cases:
            print(f"\n  [输入] '{text}'")
            processed, subtitle = self.processor.process(text)
            print(f"  [处理后] '{processed}'")
            print(f"  [字幕拼音] '{subtitle.get('pinyin', '')}'")
            print(f"  [分段] {len(subtitle.get('segments', []))} 段")


class TestEnglishPreprocessor(unittest.TestCase):
    """英文文本预处理测试"""
    
    @classmethod
    def setUpClass(cls):
        from kittentts.preprocess import TextPreprocessor
        cls.preprocessor = TextPreprocessor()
    
    def test_01_number_to_words(self):
        """测试数字转英文单词"""
        print("\n" + "=" * 60)
        print("英文预处理测试1: 数字转英文单词")
        print("=" * 60)
        
        from kittentts.preprocess import number_to_words
        
        test_cases = [
            (0, "zero"),
            (1, "one"),
            (12, "twelve"),
            (20, "twenty"),
            (100, "one hundred"),
            (123, "one hundred twenty-three"),
            (1000, "one thousand"),
            (1200, "twelve hundred"),
            (1000000, "one million"),
            (-42, "negative forty-two"),
        ]
        
        for num, expected in test_cases:
            result = number_to_words(num)
            print(f"  {num} -> {result}")
    
    def test_02_float_to_words(self):
        """测试浮点数转英文单词"""
        print("\n" + "=" * 60)
        print("英文预处理测试2: 浮点数转英文单词")
        print("=" * 60)
        
        from kittentts.preprocess import float_to_words
        
        test_cases = [
            3.14,
            -0.5,
            1.50,
            1.007,
        ]
        
        for num in test_cases:
            result = float_to_words(num)
            print(f"  {num} -> {result}")
    
    def test_03_full_preprocessor(self):
        """测试完整预处理流程"""
        print("\n" + "=" * 60)
        print("英文预处理测试3: 完整预处理流程")
        print("=" * 60)
        
        test_cases = [
            "I have 123 apples and 42 oranges.",
            "The price is $3.14.",
            "Inflation rose by 3.5% last quarter.",
            "The meeting starts at 3:30pm.",
            "Read pages 10-20 for homework.",
            "We trained a 7B parameter model.",
            "Visit https://example.com for more info.",
            "I don't know, won't you help?",
        ]
        
        for text in test_cases:
            processed = self.preprocessor(text)
            print(f"\n  输入: '{text}'")
            print(f"  输出: '{processed}'")
    
    def test_04_expand_special_cases(self):
        """测试特殊情况扩展"""
        print("\n" + "=" * 60)
        print("英文预处理测试4: 特殊情况扩展")
        print("=" * 60)
        
        from kittentts.preprocess import (
            expand_currency, expand_percentages, expand_time,
            expand_ranges, expand_ordinals, expand_units,
            expand_phone_numbers, expand_ip_addresses
        )
        
        special_tests = [
            ("货币", expand_currency, "$100, USD 1,200.50, GBP 9.99"),
            ("百分比", expand_percentages, "50% off, -2% change"),
            ("时间", expand_time, "3:30pm, 14:00, 9:05 AM"),
            ("范围", expand_ranges, "pages 10-20, years 2020-2024"),
            ("序数", expand_ordinals, "1st place, 2nd floor, 21st century"),
            ("单位", expand_units, "100km, 50kg, 25 deg C, 5GB"),
            ("电话", expand_phone_numbers, "555-1234, 1-800-555-0199"),
            ("IP地址", expand_ip_addresses, "192.168.1.1, 10.0.0.1"),
        ]
        
        for name, func, text in special_tests:
            result = func(text)
            print(f"\n  [{name}]")
            print(f"    输入: '{text}'")
            print(f"    输出: '{result}'")


class TestTextCleaner(unittest.TestCase):
    """文本清理器和音素转换测试"""
    
    @classmethod
    def setUpClass(cls):
        from kittentts.onnx_model import TextCleaner
        cls.text_cleaner = TextCleaner()
    
    def test_01_basic_tokenize(self):
        """测试基础英文分词"""
        print("\n" + "=" * 60)
        print("文本清理测试1: 基础英文分词")
        print("=" * 60)
        
        from kittentts.onnx_model import basic_english_tokenize
        
        test_cases = [
            "Hello, world!",
            "This is a test.",
            "What's your name?",
        ]
        
        for text in test_cases:
            tokens = basic_english_tokenize(text)
            print(f"  输入: '{text}'")
            print(f"  分词: {tokens}")
    
    def test_02_ensure_punctuation(self):
        """测试确保句子末尾有标点"""
        print("\n" + "=" * 60)
        print("文本清理测试2: 确保句子末尾有标点")
        print("=" * 60)
        
        from kittentts.onnx_model import ensure_punctuation
        
        test_cases = [
            "Hello world",
            "Hello world.",
            "",
            "   ",
        ]
        
        for text in test_cases:
            result = ensure_punctuation(text)
            print(f"  输入: '{text}'")
            print(f"  输出: '{result}'")
    
    def test_03_chunk_text(self):
        """测试长文本分块"""
        print("\n" + "=" * 60)
        print("文本清理测试3: 长文本分块")
        print("=" * 60)
        
        from kittentts.onnx_model import chunk_text
        
        test_cases = [
            "This is a short sentence.",
            "This is sentence one. This is sentence two. This is sentence three.",
            "A" * 500,
        ]
        
        for text in test_cases:
            chunks = chunk_text(text, max_len=100)
            print(f"\n  输入长度: {len(text)} 字符")
            print(f"  分块数量: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"    块{i+1}: '{chunk[:50]}...' (长度: {len(chunk)})")
    
    def test_04_text_cleaner(self):
        """测试文本清理器（字符到索引映射）"""
        print("\n" + "=" * 60)
        print("文本清理测试4: 文本清理器（字符到索引映射）")
        print("=" * 60)
        
        test_cases = [
            "Hello",
            "Hello, world!",
            "What's your name?",
        ]
        
        for text in test_cases:
            indices = self.text_cleaner(text)
            print(f"  输入: '{text}'")
            print(f"  索引: {indices}")
            print(f"  长度: {len(indices)}")


class TestModelLoading(unittest.TestCase):
    """模型加载测试（可选，需要网络）"""
    
    @unittest.skipUnless(False, "需要网络下载模型，跳过测试")
    def test_01_model_loading(self):
        """测试模型加载"""
        print("\n" + "=" * 60)
        print("模型加载测试1: 模型初始化")
        print("=" * 60)
        
        try:
            from kittentts import KittenTTS
            print("  [!] 开始加载模型（可能需要较长时间）...")
            model = KittenTTS("KittenML/kitten-tts-nano-0.8")
            print("  [OK] 模型加载成功!")
            print(f"  可用语音: {model.available_voices}")
        except Exception as e:
            print(f"  [ERROR] 模型加载失败: {e}")
    
    @unittest.skipUnless(False, "需要网络下载模型，跳过测试")
    def test_02_generate_audio(self):
        """测试语音生成"""
        print("\n" + "=" * 60)
        print("模型加载测试2: 语音生成")
        print("=" * 60)
        
        try:
            from kittentts import KittenTTS
            model = KittenTTS("KittenML/kitten-tts-nano-0.8")
            
            test_texts = [
                "Hello, this is a test.",
                "你好，这是一个测试。",
            ]
            
            for text in test_texts:
                print(f"\n  测试文本: '{text}'")
                audio, subtitle = model.generate_with_subtitles(
                    text=text,
                    voice='Bruno',
                    speed=1.0,
                    clean_text=True
                )
                print(f"  音频形状: {audio.shape}")
                print(f"  音频时长: {len(audio) / 24000:.2f} 秒")
                print(f"  字幕信息:")
                print(f"    原文: {subtitle.get('original', '')}")
                print(f"    拼音: {subtitle.get('pinyin', '')}")
                print(f"    语言: {subtitle.get('language', '')}")
        except Exception as e:
            print(f"  [ERROR] 语音生成失败: {e}")


def run_backend_tests():
    """运行所有后端测试"""
    print("\n" + "=" * 70)
    print("KittenTTS 后端推理服务单元测试")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestChineseProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestEnglishPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestTextCleaner))
    
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
    success = run_backend_tests()
    sys.exit(0 if success else 1)
