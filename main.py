import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse # <--- 导入 StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from uvicorn import run

# --- 配置常量 (Configuration Constants) ---
# 从环境变量读取，如果不存在则使用默认值
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://chat.z.ai/api/chat/completions")
DEFAULT_KEY = os.getenv("DEFAULT_KEY", "sk-your-key")  # 下游客户端鉴权key
UPSTREAM_TOKEN = os.getenv("UPSTREAM_TOKEN",
                           "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjMxNmJjYjQ4LWZmMmYtNGExNS04NTNkLWYyYTI5YjY3ZmYwZiIsImVtYWlsIjoiR3Vlc3QtMTc1NTg0ODU4ODc4OEBndWVzdC5jb20ifQ.PktllDySS3trlyuFpTeIZf-7hl8Qu1qYF3BxjgIul0BrNux2nX9hVzIjthLXKMWAf9V0qM8Vm_iyDqkjPGsaiQ")  # 上游API的token（回退用）
MODEL_NAME = os.getenv("MODEL_NAME", "GLM-4.5")
PORT = int(os.getenv("PORT", 8080))
# 使用 "true", "1", "t" 等不区分大小写的字符串来判断布尔值
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() in ("true", "1", "t")

# 思考内容处理策略 (Thinking Content Processing Strategy)
# "strip": 去除<details>标签; "think": 转为<think>标签; "raw": 保留原样
THINK_TAGS_MODE: Literal["strip", "think", "raw"] = "strip"

# 伪装前端头部 (Spoofed Frontend Headers)
X_FE_VERSION = "prod-fe-1.0.70"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0"
SEC_CH_UA = '"Not;A=Brand";v="99", "Microsoft Edge";v="139", "Chromium";v="139"'
SEC_CH_UA_MOB = "?0"
SEC_CH_UA_PLAT = '"Windows"'
ORIGIN_BASE = "https://chat.z.ai"

# 匿名token开关 (Anonymous Token Switch)
ANON_TOKEN_ENABLED = True

# --- 日志配置 (Logging Configuration) ---
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


# --- Pydantic 数据模型 (Data Models) ---
# 对应 Go 中的 struct
class Message(BaseModel):
    role: str
    content: str


class OpenAIRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = Field(None, alias="max_tokens")


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Delta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class Choice(BaseModel):
    index: int
    message: Optional[Message] = None
    delta: Optional[Delta] = None
    finish_reason: Optional[str] = None


class OpenAIResponse(BaseModel):
    id: str
    object: str
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "z.ai"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# --- FastAPI 应用实例 (App Instance) ---
app = FastAPI()

# 配置CORS中间件 (CORS Middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 辅助函数 (Helper Functions) ---
async def get_anonymous_token() -> Optional[str]:
    """获取匿名token，避免共享记忆"""
    headers = {
        "User-Agent": BROWSER_UA,
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "X-FE-Version": X_FE_VERSION,
        "sec-ch-ua": SEC_CH_UA,
        "sec-ch-ua-mobile": SEC_CH_UA_MOB,
        "sec-ch-ua-platform": SEC_CH_UA_PLAT,
        "Origin": ORIGIN_BASE,
        "Referer": f"{ORIGIN_BASE}/",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ORIGIN_BASE}/api/v1/auths/", headers=headers)
            if response.status_code == 200:
                data = response.json()
                token = data.get("token")
                if token:
                    logging.debug("匿名token获取成功")
                    return token
                logging.warning("匿名token为空")
                return None
            logging.warning(f"获取匿名token失败, status_code={response.status_code}")
            return None
    except Exception as e:
        logging.error(f"获取匿名token时发生异常: {e}")
        return None


def transform_thinking(s: str) -> str:
    """处理上游 'thinking' 阶段返回的内容"""
    s = re.sub(r"(?s)<summary>.*?</summary>", "", s)
    s = s.replace("</thinking>", "").replace("<Full>", "").replace("</Full>", "")
    s = s.strip()

    if THINK_TAGS_MODE == "think":
        s = re.sub(r"<details[^>]*>", "<think>", s)
        s = s.replace("</details>", "</think>")
    elif THINK_TAGS_MODE == "strip":
        s = re.sub(r"<details[^>]*>", "", s)
        s = s.replace("</details>", "")

    # 处理每行前缀 "> "
    s = s.removeprefix("> ")
    s = s.replace("\n> ", "\n")
    return s.strip()


async def stream_generator(upstream_response: httpx.Response, request_id: str) -> AsyncGenerator[str, None]:
    """从上游响应生成并转换SSE事件"""
    # 1. 发送第一个包含 role 的 chunk
    first_chunk = OpenAIResponse(
        id=request_id,
        object="chat.completion.chunk",
        model=MODEL_NAME,
        choices=[Choice(index=0, delta=Delta(role="assistant"))]
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # 2. 逐行处理上游SSE流
    async for line in upstream_response.aiter_lines():
        if not line.startswith("data:"):
            continue

        data_str = line.removeprefix("data: ").strip()
        if not data_str:
            continue

        try:
            data = json.loads(data_str)
            # 统一检查多层嵌套的错误
            error_details = (
                    data.get("error") or
                    data.get("data", {}).get("error") or
                    (data.get("data", {}).get("data", {}) or {}).get("error")
            )
            if error_details:
                logging.error(f"上游错误: code={error_details.get('code')}, detail={error_details.get('detail')}")
                break

            delta_content = data.get("data", {}).get("delta_content", "")
            phase = data.get("data", {}).get("phase", "")

            if delta_content:
                out_content = transform_thinking(delta_content) if phase == "thinking" else delta_content
                if out_content:
                    chunk = OpenAIResponse(
                        id=request_id,
                        object="chat.completion.chunk",
                        model=MODEL_NAME,
                        choices=[Choice(index=0, delta=Delta(content=out_content))]
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

            done = data.get("data", {}).get("done", False)
            if done or phase == "done":
                logging.debug("检测到流结束信号")
                break
        except json.JSONDecodeError:
            logging.warning(f"无法解析SSE数据: {data_str}")
            continue

    # 3. 发送结束 chunk 和 [DONE] 标记
    end_chunk = OpenAIResponse(
        id=request_id,
        object="chat.completion.chunk",
        model=MODEL_NAME,
        choices=[Choice(index=0, delta=Delta(), finish_reason="stop")]
    )
    yield f"data: {end_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
    logging.debug(f"流式响应完成 (request_id={request_id})")


# --- API 路由 (API Endpoints) ---
@app.options("/{path:path}")
async def handle_options(path: str):
    """处理所有OPTIONS预检请求"""
    return Response(status_code=200)


@app.get("/v1/models", response_model=ModelsResponse)
async def handle_models():
    """返回支持的模型列表"""
    return ModelsResponse(data=[ModelInfo(id=MODEL_NAME)])


@app.post("/v1/chat/completions")
async def handle_chat_completions(
        req: OpenAIRequest,
        authorization: str = Header(None)
):
    """处理聊天补全请求，支持流式和非流式"""
    # 1. 验证API Key
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    api_key = authorization.removeprefix("Bearer ")
    if api_key != DEFAULT_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    logging.debug(f"请求解析成功 - 模型: {req.model}, 流式: {req.stream}, 消息数: {len(req.messages)}")

    # 2. 构造上游请求
    chat_id = f"{int(time.time() * 1e9)}-{int(time.time())}"
    msg_id = str(int(time.time() * 1e9))

    upstream_payload: Dict[str, Any] = {
        "stream": True,  # 总是使用流式从上游获取
        "chat_id": chat_id,
        "id": msg_id,
        "model": "0727-360B-API",
        "messages": [msg.model_dump() for msg in req.messages],
        "params": {},
        "features": {"enable_thinking": True},
        "background_tasks": {"title_generation": False, "tags_generation": False},
        "mcp_servers": [],
        "model_item": {"id": "0727-360B-API", "name": "GLM-4.5", "owned_by": "openai"},
        "tool_servers": [],
        "variables": {
            "{{USER_NAME}}": "User",
            "{{USER_LOCATION}}": "Unknown",
            "{{CURRENT_DATETIME}}": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    # 3. 获取认证Token
    auth_token = UPSTREAM_TOKEN
    if ANON_TOKEN_ENABLED:
        anon_token = await get_anonymous_token()
        if anon_token:
            auth_token = anon_token
        else:
            logging.warning("匿名token获取失败，回退到固定token")

    # 4. 构造上游请求头
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": BROWSER_UA,
        "Authorization": f"Bearer {auth_token}",
        "Accept-Language": "zh-CN",
        "sec-ch-ua": SEC_CH_UA,
        "sec-ch-ua-mobile": SEC_CH_UA_MOB,
        "sec-ch-ua-platform": SEC_CH_UA_PLAT,
        "X-FE-Version": X_FE_VERSION,
        "Origin": ORIGIN_BASE,
        "Referer": f"{ORIGIN_BASE}/c/{chat_id}",
    }

    # 5. 发送请求并处理响应
    request_id = f"chatcmpl-{uuid.uuid4()}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                UPSTREAM_URL,
                json=upstream_payload,
                headers=headers,
            )
            response.raise_for_status()  # 如果状态码不是 2xx，则抛出异常
        except httpx.HTTPStatusError as e:
            logging.error(f"上游API返回错误状态: {e.response.status_code}")
            error_body = await e.response.aread()
            logging.debug(f"上游错误响应体: {error_body.decode()}")
            raise HTTPException(status_code=502, detail="Upstream API error")
        except httpx.RequestError as e:
            logging.error(f"请求上游API失败: {e}")
            raise HTTPException(status_code=502, detail="Failed to call upstream API")

        # 根据客户端请求的 stream 参数决定返回类型
        if req.stream:
            logging.debug(f"开始处理流式响应 (request_id={request_id})")
            return StreamingResponse(  # <--- 修改为 StreamingResponse
                stream_generator(response, request_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            logging.debug(f"开始处理非流式响应 (request_id={request_id})")
            full_content = ""
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line.removeprefix("data: ").strip()
                if not data_str:
                    continue
                try:
                    data = json.loads(data_str)
                    delta_content = data.get("data", {}).get("delta_content", "")
                    phase = data.get("data", {}).get("phase", "")

                    if delta_content:
                        out_content = transform_thinking(delta_content) if phase == "thinking" else delta_content
                        if out_content:
                            full_content += out_content

                    if data.get("data", {}).get("done", False) or phase == "done":
                        break
                except json.JSONDecodeError:
                    continue

            logging.debug(f"非流式响应内容收集完成, 长度: {len(full_content)}")
            return OpenAIResponse(
                id=request_id,
                object="chat.completion",
                model=MODEL_NAME,
                choices=[
                    Choice(
                        index=0,
                        message=Message(role="assistant", content=full_content),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(),  # Usage 信息未从上游获取，返回默认值
            )


# --- 主程序入口 (Main Entry Point) ---
if __name__ == "__main__":
    logging.info(f"OpenAI兼容API服务器启动在 http://0.0.0.0:{PORT}")
    logging.info(f"模型: {MODEL_NAME}")
    logging.info(f"上游: {UPSTREAM_URL}")
    logging.info(f"Debug模式: {DEBUG_MODE}")
    run(app, host="0.0.0.0", port=PORT)