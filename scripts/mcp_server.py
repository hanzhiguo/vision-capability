#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision-capability MCP 服务（stdio / JSON-RPC）。

把通用视觉能力暴露成 MCP 工具 `vision_analyze`，任何支持 MCP 的 agent 都能直接调用，
无需关心 grsai 凭证、base64、请求构造等细节。

使用方法（在 MCP 客户端配置里指向本文件，例如）：
  command: python
  args:    ["<本技能目录>/scripts/mcp_server.py"]

协议：stdio 上以换行分隔的 JSON-RPC 2.0 消息。仅实现 MCP 最小可用子集：
  - initialize / notifications/initialized
  - tools/list
  - tools/call  -> 内部 subprocess 调用 vision.py

工具 vision_analyze 入参：
  image  (string, 必填)  本地图片路径 或 http(s) URL
  mode   (string)        ocr | describe | extract | perceive（默认 perceive）
  keys   (string)        领域无关提取，逗号分隔字段名，如 "名称,材质,价格"
  prompt (string)        自定义提示词（覆盖 mode/keys 默认提示）
  model  (string)        gemini-3-flash(默认) | gemini-3-pro
  hint   (string)        附加提示，追加到结构化提取提示词
"""

import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VISION_PY = os.path.join(SCRIPT_DIR, "vision.py")

TOOLS = [
    {
        "name": "vision_analyze",
        "description": (
            "视觉能力：通过 grsai 中转站调用 Gemini 多模态模型识别图片，"
            "把图片转成纯文本/JSON，为不具备视觉的模型提供'眼睛'。"
            "支持 OCR、描述、结构化提取、面向非视觉模型的详尽感知报告(perceive)，"
            "以及按任意字段名做领域无关提取(keys)。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "本地图片路径 或 http(s) URL",
                },
                "mode": {
                    "type": "string",
                    "enum": ["ocr", "describe", "extract", "perceive"],
                    "default": "perceive",
                    "description": "识别模式；perceive 为面向非视觉模型的详尽感知报告",
                },
                "keys": {
                    "type": "string",
                    "description": "领域无关提取：逗号分隔的字段名，如 '名称,材质,价格'（优先于 mode）",
                },
                "prompt": {
                    "type": "string",
                    "description": "自定义提示词（覆盖 mode/keys 默认提示）",
                },
                "model": {
                    "type": "string",
                    "default": "gemini-3-flash",
                    "description": "Gemini 模型名",
                },
                "hint": {
                    "type": "string",
                    "description": "附加提示，追加到结构化提取提示词",
                },
            },
            "required": ["image"],
        },
    }
]


def send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def handle_initialize(req_id):
    send({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "vision-capability", "version": "1.0.0"},
        },
    })


def handle_tools_list(req_id):
    send({"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}})


def handle_tools_call(req_id, params):
    arguments = params.get("arguments", {}) or {}
    image = arguments.get("image")
    if not image:
        send({
            "jsonrpc": "2.0", "id": req_id,
            "result": {"isError": True, "content": [{"type": "text", "text": "缺少必填参数 image"}]},
        })
        return

    cmd = [sys.executable, VISION_PY, image]
    mode = arguments.get("mode", "perceive")
    if arguments.get("keys"):
        cmd += ["--keys", arguments["keys"]]
    else:
        cmd += ["--mode", mode]
    if arguments.get("prompt"):
        cmd += ["--prompt", arguments["prompt"]]
    if arguments.get("model"):
        cmd += ["--model", arguments["model"]]
    if arguments.get("hint"):
        cmd += ["--hint", arguments["hint"]]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=300)
    except Exception as e:  # noqa: BLE001
        send({
            "jsonrpc": "2.0", "id": req_id,
            "result": {"isError": True, "content": [{"type": "text", "text": f"调用失败: {e}"}]},
        })
        return

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        msg = (stderr or stdout or "未知错误")
        send({
            "jsonrpc": "2.0", "id": req_id,
            "result": {"isError": True, "content": [{"type": "text", "text": f"vision.py 返回错误: {msg}"}]},
        })
        return

    text = stdout
    if stderr:
        text = f"[stderr]\n{stderr}\n\n[result]\n{stdout}"
    send({
        "jsonrpc": "2.0", "id": req_id,
        "result": {"isError": False, "content": [{"type": "text", "text": text}]},
    })


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method")
        req_id = msg.get("id")

        if method == "initialize":
            handle_initialize(req_id)
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            handle_tools_list(req_id)
        elif method == "tools/call":
            handle_tools_call(req_id, msg.get("params", {}))
        else:
            # 对未知请求（且带 id）回个 method not found，避免客户端挂起
            if req_id is not None:
                send({
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                })


if __name__ == "__main__":
    main()
