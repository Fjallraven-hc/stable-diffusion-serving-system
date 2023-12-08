from aiohttp import web

async def handle_request(request, input_queue, output_queue):
    data = await request.text()
    input_queue.put(data)
    result = output_queue.get()  # 等待工作进程处理完毕
    return web.Response(text=result)

def start_server(input_queue, output_queue):
    app = web.Application()
    app.add_routes([web.post('/inference', lambda request: handle_request(request, input_queue, output_queue))])
    web.run_app(app)