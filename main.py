import os

from flask import Flask, render_template
from flask_cors import CORS
from flask import Flask, render_template, Response
from flask_sse import sse
import random
import time
from dotenv import load_dotenv


load_dotenv()
app = Flask(__name__, static_folder='./build', static_url_path='/')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')
@app.route('/generate')
def generate():
    def event_stream():
        count = 0
        while True:
            time.sleep(1)
            value = random.randint(1, 100)
            count += 1
            yield f'data: {value}\n\n'
            if count >= 30:
                break

    return Response(event_stream(), content_type='text/event-stream')
# 其他API路由，例如：
@app.route('/api/data')
def get_data():
    # 返回数据
    pass

print("openai_key",os.environ.get("HELLO_WORD"))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4999)
