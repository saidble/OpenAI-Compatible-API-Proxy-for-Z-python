import anthropic
import os

# --- 配置 ---
# 从环境变量读取配置，如果不存在则使用默认值
# 确保这里的 base_url 和 api_key 与 main_anthropic.py 中的配置匹配
BASE_URL = os.getenv("ANTHROPIC_API_BASE", "http://localhost:8008/v1")
API_KEY = os.getenv("ANTHROPIC_API_KEY", "sk-ant-your-key")

# 创建一个Anthropic客户端实例
try:
    client = anthropic.Anthropic(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
except TypeError as e:
    print(f"初始化 Anthropic 客户端时出错: {e}")
    print("请确保您的 anthropic 库版本是最新的 (pip install --upgrade anthropic)")
    exit(1)

# --- 对话内容 ---
# 您可以修改这里的 system prompt 和 user message 来测试不同的场景
system_prompt = "你是一个简洁的诗人，只用中文回答。"
messages = [
    {"role": "user", "content": "你好，请为我创作一首关于星空的四行诗。"}
]

# --- 方式一：非流式调用 (Non-Streaming) ---
print("--- Anthropic 非流式调用 ---")
try:
    message = client.messages.create(
        model="claude-3-opus-20240229",  # 这里的模型名会被服务器接收并回显
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
        temperature=0.7,
    )
    # Anthropic 的响应内容在 .content[0].text 中
    if message.content:
        print("AI:", message.content[0].text)
    else:
        print("AI: 未收到有效内容。")

except anthropic.APIStatusError as e:
    print(f"API 调用失败，状态码: {e.status_code}")
    print(f"响应内容: {e.response.text}")
except anthropic.APIConnectionError as e:
    print(f"无法连接到服务器: {e.__cause__}")
except Exception as e:
    print(f"发生未知错误: {e}")

print("\n" + "=" * 50 + "\n")

# --- 方式二：流式调用 (Streaming) ---
print("--- Anthropic 流式调用 ---")
try:
    print("AI: ", end="", flush=True)
    # 使用 with 上下文管理器来确保连接被正确关闭
    with client.messages.stream(
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
            model="claude-3-opus-20240229",
            temperature=0.7,
    ) as stream:
        # stream.text_stream 会逐个返回文本块
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print("\n流式传输结束。")

except anthropic.APIStatusError as e:
    print(f"API 调用失败，状态码: {e.status_code}")
    print(f"响应内容: {e.response.text}")
except anthropic.APIConnectionError as e:
    print(f"无法连接到服务器: {e.__cause__}")
except Exception as e:
    print(f"发生未知错误: {e}")