import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from webui.app import app

if __name__ == '__main__':
    print("=" * 60)
    print("[KittenTTS] WebUI 启动中...")
    print("=" * 60)
    print()
    print("请在浏览器中访问: http://localhost:5000")
    print()
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
