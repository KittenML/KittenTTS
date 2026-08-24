import urllib.request
import sys
import os

# 设置控制台编码
sys.stdout.reconfigure(encoding='utf-8')

# 首先检查本地文件
print("=" * 60)
print("Checking local index.html file")
print("=" * 60)

html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webui', 'templates', 'index.html')
print(f"File path: {html_path}")
print(f"File exists: {os.path.exists(html_path)}")

if os.path.exists(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    print(f"File size: {len(html)} characters")
    
    # 检查是否包含所有新增的元素
    checks = [
        ('themeToggle', 'theme toggle button'),
        ('file-upload-container', 'file upload container'),
        ('progress-container', 'progress bar container'),
        ('dark-mode', 'dark mode style'),
        ('initTheme', 'theme init function'),
        ('initFileUpload', 'file upload init function'),
        ('showProgress', 'progress bar show function'),
    ]
    
    all_found = True
    for check, name in checks:
        if check in html:
            print(f"[OK] Found: {name}")
        else:
            print(f"[MISSING] Not found: {name}")
            all_found = False
    
    if all_found:
        print("\n[OK] All new features are in the local file!")
    else:
        print("\n[ERROR] Some features are missing from the local file!")
else:
    print("[ERROR] Local index.html file does not exist!")

# 然后检查Flask服务返回的内容
print("\n" + "=" * 60)
print("Checking Flask server response")
print("=" * 60)

try:
    with urllib.request.urlopen('http://localhost:5000') as response:
        server_html = response.read().decode('utf-8')
        
        print(f"Server response size: {len(server_html)} characters")
        
        # 检查是否包含所有新增的元素
        all_found_server = True
        for check, name in checks:
            if check in server_html:
                print(f"[OK] Found: {name}")
            else:
                print(f"[MISSING] Not found: {name}")
                all_found_server = False
        
        if all_found_server:
            print("\n[OK] All new features are in the server response!")
        else:
            print("\n[ERROR] Some features are missing from the server response!")
            
        # 显示前1000个字符，看看返回的是什么
        print("\nFirst 1000 characters of server response:")
        print("-" * 60)
        print(server_html[:1000])
        print("-" * 60)
        
except Exception as e:
    print(f"[ERROR] Error connecting to server: {e}")
