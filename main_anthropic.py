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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from uvicorn import run

# --- 配置常量 (与之前相同) ---
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://chat.z.ai/api/chat/completions")
DEFAULT_KEY = os.getenv("ANTHROPIC_KEY", "sk-ant-your-key")
UPSTREAM_TOKEN = os.getenv("UPSTREAM_TOKEN",
                           "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjMxNmJjYjQ4LWZmMmYtNGExNS04NTNkLWYyYTI5YjY3ZmYwZiIsImVtYWlsIjoiR3Vlc3QtMTc1NTg0ODU4ODc4OEBndWVzdC5jb20ifQ.PktllDySS3trlyuFpTeIZf-7hl8Qu1qYF3BxjgIul0BrNux2nX9hVzIjthLXKMWAf9V0qM8Vm_iyDqkjPGsaiQ")
MODEL_NAME = os.getenv("MODEL_NAME", "GLM-4.5")
PORT = int(os.getenv("PORT", 8008))
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() in ("true", "1", "t")
THINK_TAGS_MODE: Literal["strip", "think", "raw"] = "strip"
ANON_TOKEN_ENABLED = True

# --- 伪装前端头部 (与之前相同) ---
X_FE_VERSION = "prod-fe-1.0.70"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0"
SEC_CH_UA = '"Not;A=Brand";v="99", "Microsoft Edge";v="139", "Chromium";v="139"'
SEC_CH_UA_MOB = "?0"
SEC_CH_UA_PLAT = '"Windows"'
ORIGIN_BASE = "https://chat.z.ai"

# --- 日志配置 (与之前相同) ---
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [ANTHROPIC] %(message)s",
)


# --- Pydantic 模型和辅助函数 (与之前相同，为完整性保留) ---
class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AnthropicRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    system: Optional[str] = None
    max_tokens: int = 1024
    stream: bool = False
    temperature: Optional[float] = None


async def get_anonymous_token() -> Optional[str]:
    headers = {
        "User-Agent": BROWSER_UA, "Accept": "*/*", "Accept-Language": "zh-CN,zh;q=0.9",
        "X-FE-Version": X_FE_VERSION, "sec-ch-ua": SEC_CH_UA, "sec-ch-ua-mobile": SEC_CH_UA_MOB,
        "sec-ch-ua-platform": SEC_CH_UA_PLAT, "Origin": ORIGIN_BASE, "Referer": f"{ORIGIN_BASE}/",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ORIGIN_BASE}/api/v1/auths/", headers=headers)
            if response.status_code == 200:
                token = response.json().get("token")
                if token: return token
    except Exception as e:
        logging.error(f"获取匿名token时发生异常: {e}")
    return None


def transform_thinking(s: str) -> str:
    s = re.sub(r"(?s)<summary>.*?</summary>", "", s)
    s = s.replace("</thinking>", "").replace("<Full>", "").replace("</Full>", "")
    s = s.strip()
    if THINK_TAGS_MODE == "strip":
        s = re.sub(r"<details[^>]*>", "", s)
        s = s.replace("</details>", "")
    s = s.removeprefix("> ")
    s = s.replace("\n> ", "\n")
    return s.strip()


async def stream_anthropic_generator(upstream_response: httpx.Response, request_id: str, requested_model: str) -> \
AsyncGenerator[str, None]:
    # 此函数逻辑是正确的，无需修改
    first_chunk = True
    usage = {"input_tokens": 0, "output_tokens": 0}
    async for line in upstream_response.aiter_lines():
        if not line.startswith("data:"): continue
        data_str = line.removeprefix("data: ").strip()
        if not data_str: continue
        try:
            data = json.loads(data_str)
            delta_content = data.get("data", {}).get("delta_content", "")
            phase = data.get("data", {}).get("phase", "")
            if first_chunk and delta_content:
                start_event = {"type": "message_start",
                               "message": {"id": request_id, "type": "message", "role": "assistant", "content": [],
                                           "model": requested_model, "stop_reason": None, "stop_sequence": None,
                                           "usage": usage}}
                yield f"event: {start_event['type']}\ndata: {json.dumps(start_event['message'])}\n\n"
                content_start_event = {"type": "content_block_start", "index": 0,
                                       "content_block": {"type": "text", "text": ""}}
                yield f"event: {content_start_event['type']}\ndata: {json.dumps(content_start_event)}\n\n"
                first_chunk = False
            if delta_content:
                out_content = transform_thinking(delta_content) if phase == "thinking" else delta_content
                if out_content:
                    usage["output_tokens"] += 1
                    delta_event = {"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "text_delta", "text": out_content}}
                    yield f"event: {delta_event['type']}\ndata: {json.dumps(delta_event)}\n\n"
            if data.get("data", {}).get("done", False) or phase == "done":
                stop_reason = "end_turn"
                content_stop_event = {"type": "content_block_stop", "index": 0}
                yield f"event: {content_stop_event['type']}\ndata: {json.dumps(content_stop_event)}\n\n"
                message_delta_event = {"type": "message_delta",
                                       "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                       "usage": {"output_tokens": usage['output_tokens']}}
                yield f"event: {message_delta_event['type']}\ndata: {json.dumps(message_delta_event)}\n\n"
                message_stop_event = {"type": "message_stop"}
                yield f"event: {message_stop_event['type']}\ndata: {json.dumps(message_stop_event)}\n\n"
                break
        except json.JSONDecodeError:
            continue


# --- FastAPI 应用实例 ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


# --- API 路由 ---
@app.post("/v1/messages")
async def handle_anthropic_message(req: AnthropicRequest, x_api_key: str = Header(None)):
    if x_api_key != DEFAULT_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    logging.info(f"收到请求 (模型: {req.model}) -> 代理到上游 (模型: {MODEL_NAME})")

    openai_messages = []
    if req.system: openai_messages.append({"role": "system", "content": req.system})
    for msg in req.messages: openai_messages.append(msg.model_dump())

    chat_id = f"{int(time.time() * 1e9)}-{int(time.time())}"

    # 【修正】补全所有上游需要的 payload 字段
    upstream_payload = {
        "stream": True,
        "chat_id": chat_id,
        "id": str(int(time.time() * 1e9)),
        "model": "0727-360B-API",
        "messages": openai_messages,
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

    auth_token = UPSTREAM_TOKEN
    if ANON_TOKEN_ENABLED:
        anon_token = await get_anonymous_token()
        if anon_token: auth_token = anon_token

    # 【修正】补全所有伪装浏览器需要的 headers
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

    request_id = f"msg_{uuid.uuid4().hex}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(UPSTREAM_URL, json=upstream_payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # 增加更详细的日志，便于调试
            error_body = await e.response.aread()
            logging.error(f"上游API返回错误状态: {e.response.status_code}, 响应: {error_body.decode()}")
            raise HTTPException(status_code=502, detail="Upstream API error")
        except httpx.RequestError as e:
            logging.error(f"请求上游API失败: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to call upstream API: {e}")

        if req.stream:
            return StreamingResponse(
                stream_anthropic_generator(response, request_id, req.model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            full_content = ""
            async for line in response.aiter_lines():
                if not line.startswith("data:"): continue
                data_str = line.removeprefix("data: ").strip()
                if not data_str: continue
                try:
                    data = json.loads(data_str)
                    delta_content = data.get("data", {}).get("delta_content", "")
                    phase = data.get("data", {}).get("phase", "")
                    if delta_content:
                        out_content = transform_thinking(delta_content) if phase == "thinking" else delta_content
                        if out_content: full_content += out_content
                    if data.get("data", {}).get("done", False) or phase == "done":
                        break
                except json.JSONDecodeError:
                    continue
            return {
                "id": request_id, "type": "message", "role": "assistant", "model": req.model,
                "content": [{"type": "text", "text": full_content}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": len(full_content) // 4}
            }


if __name__ == "__main__":
    logging.info(f"Anthropic 兼容API服务器启动在 http://0.0.0.0:{PORT}")
    logging.info(f"内部真实模型: {MODEL_NAME}")
    run(app, host="0.0.0.0", port=PORT)