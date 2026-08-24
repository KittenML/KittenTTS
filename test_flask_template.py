import sys
import os

# 添加项目路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置Flask应用的模板目录
from flask import Flask, render_template

# 创建一个测试Flask应用
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webui', 'templates'))

# 禁用模板缓存
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

print("=" * 60)
print("Testing Flask template loading")
print("=" * 60)

# 打印模板目录
print(f"Template folder: {app.template_folder}")
print(f"Template folder exists: {os.path.exists(app.template_folder)}")

# 检查index.html文件
index_path = os.path.join(app.template_folder, 'index.html')
print(f"Index.html path: {index_path}")
print(f"Index.html exists: {os.path.exists(index_path)}")

if os.path.exists(index_path):
    with open(index_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
    print(f"File size: {len(file_content)} characters")
    
    # 检查是否包含新增功能
    checks = [
        ('themeToggle', 'theme toggle button'),
        ('file-upload-container', 'file upload container'),
        ('progress-container', 'progress bar container'),
        ('dark-mode', 'dark mode style'),
        ('initTheme', 'theme init function'),
        ('initFileUpload', 'file upload init function'),
        ('showProgress', 'progress bar show function'),
    ]
    
    print("\nChecking file content:")
    all_found = True
    for check, name in checks:
        if check in file_content:
            print(f"[OK] Found: {name}")
        else:
            print(f"[MISSING] Not found: {name}")
            all_found = False
    
    if all_found:
        print("\n[OK] All new features are in the file!")
    else:
        print("\n[ERROR] Some features are missing from the file!")
else:
    print("[ERROR] index.html file does not exist!")

# 现在测试Flask的render_template函数
print("\n" + "=" * 60)
print("Testing render_template function")
print("=" * 60)

with app.app_context():
    try:
        rendered = render_template('index.html')
        print(f"Rendered template size: {len(rendered)} characters")
        
        print("\nChecking rendered content:")
        all_found_rendered = True
        for check, name in checks:
            if check in rendered:
                print(f"[OK] Found: {name}")
            else:
                print(f"[MISSING] Not found: {name}")
                all_found_rendered = False
        
        if all_found_rendered:
            print("\n[OK] All new features are in the rendered template!")
        else:
            print("\n[ERROR] Some features are missing from the rendered template!")
            
            # 显示渲染内容的前1000个字符
            print("\nFirst 1000 characters of rendered content:")
            print("-" * 60)
            print(rendered[:1000])
            print("-" * 60)
            
    except Exception as e:
        print(f"[ERROR] Error rendering template: {e}")
        import traceback
        traceback.print_exc()

# 比较文件内容和渲染内容
if os.path.exists(index_path):
    with open(index_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
    
    with app.app_context():
        try:
            rendered = render_template('index.html')
            
            print("\n" + "=" * 60)
            print("Comparing file content and rendered content")
            print("=" * 60)
            
            if file_content == rendered:
                print("[OK] File content and rendered content are identical!")
            else:
                print("[ERROR] File content and rendered content are different!")
                print(f"File size: {len(file_content)}")
                print(f"Rendered size: {len(rendered)}")
                
                # 找出差异
                if len(file_content) != len(rendered):
                    print(f"\nSize difference: {abs(len(file_content) - len(rendered))} characters")
                    
                    # 检查前100个字符是否相同
                    min_len = min(len(file_content), len(rendered))
                    for i in range(min_len):
                        if file_content[i] != rendered[i]:
                            print(f"\nFirst difference at position {i}")
                            print(f"File: {file_content[max(0, i-20):i+20]}")
                            print(f"Rendered: {rendered[max(0, i-20):i+20]}")
                            break
                            
        except Exception as e:
            print(f"[ERROR] Error comparing: {e}")
