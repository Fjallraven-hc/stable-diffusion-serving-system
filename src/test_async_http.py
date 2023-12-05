import asyncio
import multiprocessing
from aiohttp import web
import json

def worker(input_queue, output_queue):
    while True:
        # 接收请求数据
        data = input_queue.get()
        if data is None:
            break  # 结束信号

        # 模拟数据处理
        result = {"status": "processed", "original": data}

        # 发送处理结果
        output_queue.put(result)

async def handle_request(request):
    data = await request.json()

    # 发送数据到工作队列
    input_queue.put(data)

    # 从结果队列获取响应
    result = await asyncio.get_running_loop().run_in_executor(None, output_queue.get)

    # 返回HTTP响应
    return web.Response(text=json.dumps(result))

if __name__ == "__main__":
    try:
        # 创建队列
        input_queue = multiprocessing.Queue()
        output_queue = multiprocessing.Queue()

        # 创建工作进程
        worker_process = multiprocessing.Process(target=worker, args=(input_queue, output_queue))
        worker_process.start()

        # 设置异步HTTP服务器
        app = web.Application()
        app.router.add_post('/inference', handle_request)
        loop = asyncio.get_event_loop()

        # 运行HTTP服务器
        web.run_app(app, port=8080)
    except KeyboardInterrupt:
        print("-"*10,"Main process received KeyboardInterrupt","-"*10)