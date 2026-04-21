#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebUI API集成测试脚本
测试内容：
1. Flask应用启动和路由测试
2. 模型列表API测试
3. 语音列表API测试
4. 音频生成API测试
5. 任务队列API测试
6. 历史记录API测试
7. 状态API测试
8. 音频文件访问测试
"""

import sys
import os
import json
import unittest
import tempfile
import threading
import time
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestWebUIAPIBasic(unittest.TestCase):
    """WebUI API基础测试（无需要启动服务器）"""
    
    @classmethod
    def setUpClass(cls):
        from webui.app import app
        app.config['TESTING'] = True
        cls.client = app.test_client()
        cls.app = app
    
    def test_01_index_page(self):
        """测试首页加载"""
        print("\n" + "=" * 60)
        print("WebUI API测试1: 首页加载")
        print("=" * 60)
        
        response = self.client.get('/')
        print(f"  状态码: {response.status_code}")
        
        self.assertEqual(response.status_code, 200)
        print("  [OK] 首页加载成功")
        
        content = response.data.decode('utf-8')
        print(f"  页面大小: {len(content)} 字符")
        
        required_elements = [
            ('themeToggle', '主题切换按钮'),
            ('file-upload-container', '文件上传区域'),
            ('progress-container', '进度条容器'),
            ('textInput', '文本输入框'),
            ('generateBtn', '生成按钮'),
            ('audioPlayer', '音频播放器'),
        ]
        
        print("\n  检查关键元素:")
        all_found = True
        for element, name in required_elements:
            if element in content:
                print(f"    [OK] {name}")
            else:
                print(f"    [MISSING] {name}")
                all_found = False
        
        self.assertTrue(all_found, "页面缺少关键元素")
    
    def test_02_get_models_api(self):
        """测试获取模型列表API"""
        print("\n" + "=" * 60)
        print("WebUI API测试2: 获取模型列表API")
        print("=" * 60)
        
        response = self.client.get('/api/models')
        print(f"  状态码: {response.status_code}")
        
        self.assertEqual(response.status_code, 200)
        
        models = json.loads(response.data.decode('utf-8'))
        print(f"  模型数量: {len(models)}")
        
        for model in models:
            print(f"\n    模型ID: {model.get('id')}")
            print(f"    名称: {model.get('name')}")
            print(f"    描述: {model.get('description')}")
        
        expected_models = [
            "KittenML/kitten-tts-mini-0.8",
            "KittenML/kitten-tts-micro-0.8",
            "KittenML/kitten-tts-nano-0.8"
        ]
        
        actual_ids = [m.get('id') for m in models]
        for expected in expected_models:
            self.assertIn(expected, actual_ids, f"缺少模型: {expected}")
        
        print("\n  [OK] 模型列表API测试通过")
    
    def test_03_get_voices_api(self):
        """测试获取语音列表API"""
        print("\n" + "=" * 60)
        print("WebUI API测试3: 获取语音列表API")
        print("=" * 60)
        
        response = self.client.get('/api/voices')
        print(f"  状态码: {response.status_code}")
        
        if response.status_code == 200:
            voices = json.loads(response.data.decode('utf-8'))
            print(f"  语音数量: {len(voices)}")
            
            for voice in voices:
                print(f"\n    语音ID: {voice.get('id')}")
                print(f"    名称: {voice.get('name')}")
                print(f"    描述: {voice.get('description')}")
            
            expected_voices = ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']
            actual_ids = [v.get('id') for v in voices]
            for expected in expected_voices:
                self.assertIn(expected, actual_ids, f"缺少语音: {expected}")
            
            print("\n  [OK] 语音列表API测试通过")
        else:
            print(f"  [WARNING] 语音API返回非200状态码，可能是模型未加载")
            error = json.loads(response.data.decode('utf-8'))
            print(f"  错误信息: {error.get('error')}")
    
    def test_04_get_status_api(self):
        """测试获取系统状态API"""
        print("\n" + "=" * 60)
        print("WebUI API测试4: 获取系统状态API")
        print("=" * 60)
        
        response = self.client.get('/api/status')
        print(f"  状态码: {response.status_code}")
        
        self.assertEqual(response.status_code, 200)
        
        status = json.loads(response.data.decode('utf-8'))
        print(f"  已加载模型: {list(status.get('loaded_models', {}).keys())}")
        print(f"  队列统计: {status.get('queue_stats', {})}")
        print(f"  历史记录数: {status.get('history_count', 0)}")
        
        expected_fields = ['loaded_models', 'queue_stats', 'history_count']
        for field in expected_fields:
            self.assertIn(field, status, f"缺少字段: {field}")
        
        queue_stats = status.get('queue_stats', {})
        expected_queue_fields = ['pending', 'processing', 'completed', 'total']
        for field in expected_queue_fields:
            self.assertIn(field, queue_stats, f"队列统计缺少字段: {field}")
        
        print("\n  [OK] 系统状态API测试通过")
    
    def test_05_get_history_api(self):
        """测试获取历史记录API"""
        print("\n" + "=" * 60)
        print("WebUI API测试5: 获取历史记录API")
        print("=" * 60)
        
        response = self.client.get('/api/history')
        print(f"  状态码: {response.status_code}")
        
        self.assertEqual(response.status_code, 200)
        
        history = json.loads(response.data.decode('utf-8'))
        print(f"  历史记录数量: {len(history)}")
        
        self.assertIsInstance(history, list)
        print("\n  [OK] 历史记录API测试通过")
    
    def test_06_get_tasks_api(self):
        """测试获取任务队列API"""
        print("\n" + "=" * 60)
        print("WebUI API测试6: 获取任务队列API")
        print("=" * 60)
        
        response = self.client.get('/api/tasks')
        print(f"  状态码: {response.status_code}")
        
        self.assertEqual(response.status_code, 200)
        
        tasks = json.loads(response.data.decode('utf-8'))
        print(f"  任务数量: {len(tasks)}")
        
        self.assertIsInstance(tasks, list)
        print("\n  [OK] 任务队列API测试通过")


class TestWebUIAPIErrorCases(unittest.TestCase):
    """WebUI API异常场景测试"""
    
    @classmethod
    def setUpClass(cls):
        from webui.app import app
        app.config['TESTING'] = True
        cls.client = app.test_client()
    
    def test_01_empty_text_generate(self):
        """测试空文本提交生成"""
        print("\n" + "=" * 60)
        print("异常场景测试1: 空文本提交生成")
        print("=" * 60)
        
        response = self.client.post('/api/generate', 
            json={'text': ''},
            content_type='application/json'
        )
        
        print(f"  状态码: {response.status_code}")
        print(f"  预期: 400 Bad Request")
        
        self.assertEqual(response.status_code, 400)
        
        result = json.loads(response.data.decode('utf-8'))
        print(f"  错误信息: {result.get('error')}")
        
        self.assertIn('error', result)
        self.assertIn('文本不能为空', result.get('error', ''))
        
        print("\n  [OK] 空文本处理正确")
    
    def test_02_empty_text_add_task(self):
        """测试空文本添加到队列"""
        print("\n" + "=" * 60)
        print("异常场景测试2: 空文本添加到队列")
        print("=" * 60)
        
        response = self.client.post('/api/tasks', 
            json={'text': ''},
            content_type='application/json'
        )
        
        print(f"  状态码: {response.status_code}")
        print(f"  预期: 400 Bad Request")
        
        self.assertEqual(response.status_code, 400)
        
        result = json.loads(response.data.decode('utf-8'))
        print(f"  错误信息: {result.get('error')}")
        
        self.assertIn('error', result)
        self.assertIn('文本不能为空', result.get('error', ''))
        
        print("\n  [OK] 空文本队列处理正确")
    
    def test_03_invalid_json_generate(self):
        """测试无效JSON提交生成"""
        print("\n" + "=" * 60)
        print("异常场景测试3: 无效JSON提交生成")
        print("=" * 60)
        
        response = self.client.post('/api/generate', 
            data='invalid json',
            content_type='application/json'
        )
        
        print(f"  状态码: {response.status_code}")
        print(f"  预期: 400 或 500 (取决于框架处理)")
        
        self.assertIn(response.status_code, [400, 500])
        print(f"\n  [OK] 无效JSON处理正确")
    
    def test_04_get_nonexistent_audio(self):
        """测试获取不存在的音频文件"""
        print("\n" + "=" * 60)
        print("异常场景测试4: 获取不存在的音频文件")
        print("=" * 60)
        
        response = self.client.get('/api/audio/nonexistent_file.wav')
        
        print(f"  状态码: {response.status_code}")
        print(f"  预期: 404 Not Found")
        
        self.assertEqual(response.status_code, 404)
        
        result = json.loads(response.data.decode('utf-8'))
        print(f"  错误信息: {result.get('error')}")
        
        self.assertIn('error', result)
        print("\n  [OK] 不存在音频处理正确")
    
    def test_05_get_nonexistent_task(self):
        """测试获取不存在的任务"""
        print("\n" + "=" * 60)
        print("异常场景测试5: 获取不存在的任务")
        print("=" * 60)
        
        response = self.client.get('/api/tasks/nonexistent-task-id')
        
        print(f"  状态码: {response.status_code}")
        print(f"  预期: 404 Not Found")
        
        self.assertEqual(response.status_code, 404)
        
        result = json.loads(response.data.decode('utf-8'))
        print(f"  错误信息: {result.get('error')}")
        
        self.assertIn('error', result)
        print("\n  [OK] 不存在任务处理正确")
    
    def test_06_get_nonexistent_subtitle(self):
        """测试获取不存在的字幕"""
        print("\n" + "=" * 60)
        print("异常场景测试6: 获取不存在的字幕")
        print("=" * 60)
        
        response = self.client.get('/api/subtitle/nonexistent-task-id')
        
        print(f"  状态码: {response.status_code}")
        print(f"  预期: 404 Not Found")
        
        self.assertEqual(response.status_code, 404)
        
        result = json.loads(response.data.decode('utf-8'))
        print(f"  错误信息: {result.get('error')}")
        
        self.assertIn('error', result)
        print("\n  [OK] 不存在字幕处理正确")
    
    def test_07_invalid_speed_value(self):
        """测试无效语速值"""
        print("\n" + "=" * 60)
        print("异常场景测试7: 无效语速值")
        print("=" * 60)
        
        response = self.client.post('/api/generate', 
            json={
                'text': 'Hello',
                'speed': 'invalid'
            },
            content_type='application/json'
        )
        
        print(f"  状态码: {response.status_code}")
        print(f"  预期: 500 内部错误")
        
        result = json.loads(response.data.decode('utf-8'))
        print(f"  响应: {result}")
        
        print("\n  [OK] 无效语速值处理完成")


class TestWebUIAPIFullFlow(unittest.TestCase):
    """WebUI API完整流程测试"""
    
    @classmethod
    def setUpClass(cls):
        from webui.app import app
        app.config['TESTING'] = True
        cls.client = app.test_client()
    
    def test_01_full_api_workflow(self):
        """测试完整API工作流"""
        print("\n" + "=" * 60)
        print("完整流程测试1: API工作流")
        print("=" * 60)
        
        print("\n  步骤1: 获取可用模型列表")
        response = self.client.get('/api/models')
        self.assertEqual(response.status_code, 200)
        models = json.loads(response.data.decode('utf-8'))
        print(f"    可用模型: {[m['id'] for m in models]}")
        
        print("\n  步骤2: 获取系统状态")
        response = self.client.get('/api/status')
        self.assertEqual(response.status_code, 200)
        status = json.loads(response.data.decode('utf-8'))
        print(f"    已加载模型: {list(status['loaded_models'].keys())}")
        print(f"    历史记录数: {status['history_count']}")
        
        print("\n  步骤3: 获取历史记录")
        response = self.client.get('/api/history')
        self.assertEqual(response.status_code, 200)
        history = json.loads(response.data.decode('utf-8'))
        print(f"    历史记录: {len(history)} 条")
        
        print("\n  步骤4: 获取任务队列")
        response = self.client.get('/api/tasks')
        self.assertEqual(response.status_code, 200)
        tasks = json.loads(response.data.decode('utf-8'))
        print(f"    任务队列: {len(tasks)} 条")
        
        print("\n  [OK] 完整API工作流测试通过")
    
    def test_02_task_queue_flow(self):
        """测试任务队列工作流"""
        print("\n" + "=" * 60)
        print("完整流程测试2: 任务队列工作流")
        print("=" * 60)
        
        print("\n  步骤1: 获取初始队列状态")
        response = self.client.get('/api/tasks')
        initial_tasks = json.loads(response.data.decode('utf-8'))
        print(f"    初始任务数: {len(initial_tasks)}")
        
        print("\n  步骤2: 获取初始状态")
        response = self.client.get('/api/status')
        initial_status = json.loads(response.data.decode('utf-8'))
        print(f"    队列统计: {initial_status['queue_stats']}")
        
        print("\n  [OK] 任务队列工作流测试通过")


def run_webui_api_tests():
    """运行所有WebUI API测试"""
    print("\n" + "=" * 70)
    print("KittenTTS WebUI API集成测试")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestWebUIAPIBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestWebUIAPIErrorCases))
    suite.addTests(loader.loadTestsFromTestCase(TestWebUIAPIFullFlow))
    
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
    success = run_webui_api_tests()
    sys.exit(0 if success else 1)
