import openai

# --- 配置 ---
# 创建一个OpenAI客户端实例
# 重要：
# 1. base_url 必须指向您本地运行的FastAPI服务的地址
# 2. api_key 必须是您在 main.py 中设置的 DEFAULT_KEY
client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-your-key"
)

# --- 对话内容 ---
# 模拟一个多轮对话的上下文
messages = [
    {"role": "system", "content": "你是一个乐于助人的AI助手。"},
    {"role": "user", "content": "你好，请问你叫什么名字？"},
    {"role": "assistant", "content": "我是一个大型语言模型，你可以叫我小智。"},
    {"role": "user", "content": "好的小智，请帮我写一首关于秋天的五言绝句。"}
]

# --- 方式一：非流式调用 (Non-Streaming) ---
# 一次性获取完整的回答
print("--- 非流式调用 (等待完整响应) ---")
try:
    response = client.chat.completions.create(
        model="GLM-4.5",  # 这里的模型名称需要与您服务中定义的 MODEL_NAME 一致
        messages=messages,
        stream=False,
        temperature=0.7,
    )
    # 打印完整的回答内容
    print("AI:", response.choices[0].message.content)

except openai.APIError as e:
    print(f"API 调用失败: {e}")

print("\n" + "=" * 50 + "\n")

# --- 方式二：流式调用 (Streaming) ---
# 逐字或逐词接收回答，实现打字机效果
print("--- 流式调用 (实时打印响应) ---")
try:
    # 发起流式请求
    stream_response = client.chat.completions.create(
        model="GLM-4.5",
        messages=messages,
        stream=True,
        temperature=0.7,
    )

    print("AI: ", end="", flush=True)
    # 遍历从服务器返回的数据块
    for chunk in stream_response:
        # 检查是否有内容，并打印
        if chunk.choices[0].delta and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)

    print("\n流式传输结束。")

except openai.APIError as e:
    print(f"API 调用失败: {e}")