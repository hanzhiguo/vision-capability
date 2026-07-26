#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision-capability — 通用「视觉能力」，可独立抽取、与非视觉模型配合。

通过 grsai 第三方中转站调用 Google Gemini 多模态模型，把图片转成纯文本/JSON，
让任何不会看图的语言模型获得"眼睛"。本脚本**不含任何业务领域内容**（如商品/SKU），
只提供通用的视觉转述能力：

  - ocr      精确转录图中文字（表格保留行列）
  - describe 自然语言描述图片内容
  - extract  把图片信息抽成 JSON 结构（自动推断字段）
  - perceive 为纯文本模型定制的详尽感知报告（概述/OCR/对象/布局/颜色/数据/需核验）
  - keys     指定任意字段名做结构化提取（领域无关，例如 --keys 名称,材质,价格）
  - function 调用 functions.json 里用户自定义的功能化提取（默认无，可自建）

复用的调用链路（来自 GRSAI 中转站）：
  - 端点:  {GRSAI_CHAT_BASE_URL}/v1/chat/completions   (OpenAI 兼容)
  - 鉴权:  Authorization: Bearer <GRSAI_API_KEY>
  - 模型:  gemini-3-flash (默认) / gemini-3-pro
  - 图片:  以 OpenAI 视觉消息格式 image_url 传入（本地图自动 base64 成 data URL）

仅依赖 Python 标准库，无需 pip install。凭证只从环境变量或本地 config 读取，不外泄。
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://grsaiapi.com"
DEFAULT_MODEL = "gemini-3-flash"

SUPPORTED_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".pdf", ".tif", ".tiff"
}

AGENT_PROMPT = r"""你是一名专业的「视觉理解与结构化分析智能体（Vision Understanding Agent）」。

你的任务：
分析用户上传的图片，并将图片中的视觉信息转换为结构化数据。

你的输出不是给人看的图片描述，而是给另一个文本大语言模型使用的视觉知识。
文本模型无法直接看到图片，它只能依赖你的分析结果进行：
- 图片理解
- 内容生成
- UI复刻
- HTML/CSS生成
- 设计分析
- 产品营销分析
- 图片生成提示词设计
- 知识库检索

========================
核心分析原则
========================

1. 先判断图片类型

必须从以下类型选择：
- pure_text: 纯文字、截图、文档、说明书、PDF页面
- ui_design: 网站、APP、后台系统、软件界面截图
- poster: 海报、电商详情页、广告图、营销图片
- product: 产品摄影、商品图片、工业设计图片
- scene: 场景照片、环境图片
- diagram: 流程图、结构图、技术图
- unknown: 无法判断


2. 不要只描述"看到了什么"

需要回答：
- 图片由哪些元素组成？
- 元素在哪里？
- 元素之间是什么关系？
- 使用了什么设计规律？
- 如何复现这个图片？


3. 禁止幻觉

如果无法确认：使用 "unknown" 或者 "无法确认"
不要编造：品牌、产品型号、人物身份、隐藏文字


========================
分析维度
========================


一、基础信息
分析：图片类型、图片主题、使用场景、视觉目的

二、视觉布局
分析：整体构图、上中下结构、左右结构、主视觉位置、阅读顺序

三、元素识别
识别：图片元素、文本元素、UI组件、图标、装饰元素、背景
每个元素包含：名称、类型、位置、尺寸比例、视觉作用

四、文字分析
识别图片中的所有文字。包括：原始文本、标题、副标题、标签、按钮文字
分析：字体大小、粗细、对齐方式、颜色、层级

五、视觉风格
分析：色彩体系、光影、材质、风格关键词
例如：minimal, luxury, technology, industrial, modern, premium

六、空间关系
描述元素之间的位置关系。例如：product: center, title: above product, button: bottom-right


========================
不同图片类型专项分析
========================



如果 image_type = pure_text


重点输出：
- OCR文本
- 标题层级
- 段落结构
- 表格结构
- 文档布局

用于：OCR、翻译、重新排版



------------------------


如果 image_type = ui_design


重点分析：

页面结构：Header, Hero, Sidebar, Content Area, Footer
组件：Button, Card, Input, Navigation, Modal
输出：组件名称、尺寸、间距、圆角、阴影、颜色

目标：可以根据结果重新生成 HTML / React / Vue / CSS



------------------------


如果 image_type = poster


重点分析：

广告结构：主标题、产品主体、卖点、CTA按钮、装饰元素
分析：营销逻辑、视觉焦点、情绪表达、摄影风格

目标：可以生成类似海报。



------------------------


如果 image_type = product


重点分析：
产品：外观、材质、结构、颜色、关键零件
摄影：镜头角度、景别、光线、背景、景深

目标：用于产品建模、AI生成、商品详情页



========================
输出格式
========================


必须返回 JSON。格式：

{
  "image_type": "",
  "summary": "",
  "visual_purpose": "",
  "layout": {
    "structure": "",
    "reading_order": [],
    "main_focus": ""
  },
  "elements": [
    {
      "name": "",
      "type": "",
      "description": "",
      "position": "",
      "importance": ""
    }
  ],
  "text_content": [
    {
      "text": "",
      "role": "",
      "style": ""
    }
  ],
  "style_analysis": {
    "visual_style": "",
    "colors": [],
    "lighting": "",
    "material": ""
  },
  "spatial_relationships": [
    {
      "object1": "",
      "object2": "",
      "relationship": ""
    }
  ],
  "reconstruction_hint": "",
  "ai_generation_keywords": [],
  "confidence": 0
}


========================
最终目标
========================

你的输出应该让一个没有看到图片的文本模型，像"看过图片一样"理解图片。
优先提供：结构信息 > 视觉关系 > 设计规律 > 文字内容 > 主观评价。
不要写散文式描述。输出必须结构化、准确、可计算。"""

# 不同模式下的默认提示词（用户可用 --prompt 覆盖）
DEFAULT_PROMPTS = {"agent": AGENT_PROMPT}


# ----------------------------------------------------------------------------
# 凭证读取：环境变量优先，其次本地 config（不写日志、不回显）
# ----------------------------------------------------------------------------
def load_api_key():
    key = os.environ.get("GRSAI_API_KEY")
    if key and key.strip():
        return key.strip()
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.txt"),
        os.path.expanduser("~/.workbuddy/skills/vision-capability/config.txt"),
        ".grsai_key",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() in ("GRSAI_API_KEY", "API_KEY"):
                                return v.strip()
                        else:
                            return line
            except OSError:
                continue
    return None


def load_base_url():
    return (
        os.environ.get("GRSAI_CHAT_BASE_URL")
        or os.environ.get("GRSAI_BASE_URL")
        or DEFAULT_BASE_URL
    ).rstrip("/")


def load_agent_prompt(override=None):
    """视觉理解智能体（agent 模式）的提示词。

    优先级：用户 --prompt 覆盖 > 同目录 agent_prompt.txt > 内置 AGENT_PROMPT 默认。
    """
    if override:
        return override
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_prompt.txt")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            if txt:
                return txt
        except OSError:
            pass
    return AGENT_PROMPT


# ----------------------------------------------------------------------------
# 用户自定义功能（functions.json，默认空；可建任意领域功能，与脚本解耦）
# ----------------------------------------------------------------------------
def load_functions():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "functions.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError) as e:
            sys.stderr.write(f"[warn] 读取 functions.json 失败，使用空配置: {e}\n")
    return {}


def build_function_prompt(func, extra=""):
    """由功能定义生成严格提取提示词（functions.json 可用 prompt 覆盖）。"""
    if func.get("prompt"):
        prompt = func["prompt"]
    else:
        fields = func.get("fields", [])
        categories = func.get("categories") or []
        has_struct = any(f.get("struct") for f in fields)
        field_lines = "\n".join(f'- "{f["key"]}": {f.get("desc", "")}' for f in fields)
        prompt = (
            "你是一个严格的数据提取助手。请从图片中识别并提取以下字段，"
            "仅以 JSON 对象返回，不要任何额外说明，也不要用 Markdown 代码块包裹。\n\n"
            "字段列表（必须全部包含这些键，顺序无所谓）：\n" + field_lines
        )
        if has_struct:
            prompt += (
                "\n\n结构化字段说明：\n"
                "- 含数组型字段时，每个元素为 {\"键\": 值} 形式。"
            )
        if categories:
            cat_list = json.dumps(categories, ensure_ascii=False)
            prompt += (
                f"\n\n类目必须严格从以下列表中选择一个最贴切的，不得新建或改写：\n{cat_list}\n"
                f"若无法确定，选择列表最后一个并在值后追加\"（核验）\"。"
            )
        prompt += (
            "\n\n重要规则：\n"
            "1. 不虚构：图片中确实没有的字段，值填 null（绝不用推测值填充）。\n"
            "2. 不确定：保留最佳判断值，并在值后追加\"（核验）\"，例如 \"1:35（核验）\"。\n"
            "3. 价格/重量/数量等只填图片明示内容，不换算或估算。\n"
            "4. 只输出一个 JSON 对象，键名必须与上面完全一致。"
        )
    if extra:
        prompt += "\n\n补充提示（务必遵守）：\n" + extra
    return prompt


def build_keys_prompt(keys, extra=""):
    """领域无关的「按字段名提取」：用户传任意键列表，返回 JSON。"""
    clean = [k.strip() for k in keys if k.strip()]
    if not clean:
        raise ValueError("--keys 至少需要一个字段名")
    field_lines = "\n".join(f'- "{k}"' for k in clean)
    prompt = (
        "你是一个严格的数据提取助手。请从图片中识别并提取以下字段，"
        "仅以 JSON 对象返回，不要任何额外说明，也不要用 Markdown 代码块包裹。\n\n"
        "字段列表（必须全部包含这些键）：\n" + field_lines +
        "\n\n重要规则：\n"
        "1. 不虚构：图片中确实没有的字段，值填 null。\n"
        "2. 不确定：保留最佳判断值并在其后追加\"（核验）\"。\n"
        "3. 价格/重量/数量等只填图片明示内容，不换算估算。\n"
        "4. 只输出一个 JSON 对象，键名必须与上面完全一致。"
    )
    if extra:
        prompt += "\n\n补充提示：\n" + extra
    return prompt


# ----------------------------------------------------------------------------
# 图片 -> 可发送内容
# ----------------------------------------------------------------------------
def image_to_content(src, dry=False):
    """本地路径编码为 data URL；http(s) URL 直接使用。dry 模式不读文件。"""
    if dry:
        return {"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,<DRY-RUN:{src}>"}}
    if src.startswith("http://") or src.startswith("https://") or src.startswith("data:"):
        return {"type": "image_url", "image_url": {"url": src}}

    if not os.path.exists(src):
        raise FileNotFoundError(f"图片文件不存在: {src}")
    ext = os.path.splitext(src)[1].lower()
    if ext and ext not in SUPPORTED_EXT:
        sys.stderr.write(f"[warn] 未识别的图片后缀 {ext}，仍尝试发送\n")
    size = os.path.getsize(src)
    if size > 20 * 1024 * 1024:
        raise ValueError(f"图片过大 ({size} bytes > 20MB 上限)")

    mime = mimetypes.guess_type(src)[0] or "image/png"
    with open(src, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def build_messages(prompt, images, system=None, dry=False):
    content = [{"type": "text", "text": prompt}]
    for src in images:
        content.append(image_to_content(src, dry=dry))
    user_msg = {"role": "user", "content": content}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(user_msg)
    return messages


# ----------------------------------------------------------------------------
# 调用中转站
# ----------------------------------------------------------------------------
def call_relay(base_url, api_key, model, messages, timeout=180):
    url = base_url + "/v1/chat/completions"
    payload = {"model": model, "stream": False, "messages": messages}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError(f"HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"网络错误: {e.reason}")

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        raise RuntimeError(f"返回非 JSON: {body[:2000]}")


def extract_text(resp):
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return json.dumps(resp, ensure_ascii=False, indent=2)


def parse_json_object(text):
    """尽量从模型输出中解析出 JSON（兼容 ```json 围栏 / 前后多余文字）。"""
    if text is None:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
        t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(t[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    funcs = load_functions()

    ap = argparse.ArgumentParser(
        description="通用视觉能力：通过 grsai 中转站调用 Gemini 多模态模型识别图片"
                    "（OCR/描述/提取/感知/按字段提取），为非视觉模型提供眼睛。"
    )
    ap.add_argument("images", nargs="+", help="本地图片路径 或 http(s) 图片 URL，可多个")
    ap.add_argument("--prompt", help="自定义提示词（覆盖模式默认提示）")
    ap.add_argument(
        "--mode",
        choices=["agent"],
        default="agent",
        help="预设模式：agent=视觉理解智能体(最终版,固定JSON Schema,唯一模式)",
    )
    ap.add_argument("--keys", help="领域无关提取：逗号分隔的字段名，如 名称,材质,价格（返回 JSON）")
    ap.add_argument(
        "--function",
        help="调用 functions.json 里自定义的功能（如 1）；默认无，需自建",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"模型 (默认 {DEFAULT_MODEL})")
    ap.add_argument("--hint", help="额外提示文本，追加到 keys/function 提示词")
    ap.add_argument("--system", help="可选 system prompt")
    ap.add_argument("--base-url", help="中转站 base url")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出原始响应")
    ap.add_argument("--output", help="将结果写入该 JSON 文件路径（结构化模式）")
    ap.add_argument("--per-image", action="store_true",
                    help="结构化提取时每张图单独请求，结果聚合为 JSON 数组（适合批量）")
    ap.add_argument("--timeout", type=int, default=180, help="请求超时秒数")
    ap.add_argument("--dry-run", action="store_true",
                    help="只构造并打印请求 payload，不真正发起网络请求（无需 API Key）")
    args = ap.parse_args()

    # 判定走哪条提取路径
    func_key = args.function
    keys = [k for k in (args.keys or "").split(",")] if args.keys else []
    structured = True  # agent 是唯一模式，始终结构化 JSON 输出

    base_url = (args.base_url or load_base_url()).rstrip("/")

    # ---- 构造 prompt / system ----
    if func_key is not None:
        func = funcs.get(func_key)
        if not func:
            sys.stderr.write(
                f"未找到功能 {func_key}。请在脚本同目录 functions.json 中定义它，"
                "或改用 --keys 做领域无关提取。\n"
            )
            sys.exit(2)
        prompt = args.prompt or build_function_prompt(func, extra=args.hint or "")
        system = args.system if args.system else "你只输出 JSON 数据，不输出任何解释性文字。"
    elif keys:
        if len(keys) == 1 and not keys[0].strip():
            sys.stderr.write("错误：--keys 至少需要一个字段名\n")
            sys.exit(2)
        prompt = args.prompt or build_keys_prompt(keys, extra=args.hint or "")
        system = args.system if args.system else "你只输出 JSON 数据，不输出任何解释性文字。"
    elif args.mode == "agent":
        prompt = load_agent_prompt(args.prompt)
        system = args.system if args.system else "你只输出 JSON 数据，不输出任何解释性文字。返回的 JSON 必须严格符合指定的格式。"
    else:
        prompt = args.prompt or DEFAULT_PROMPTS[args.mode]
        system = args.system

    # ---- dry-run ----
    if args.dry_run:
        if structured and args.per_image and len(args.images) > 1:
            payloads = [
                {"model": args.model, "stream": False,
                 "messages": build_messages(prompt, [src], system, dry=True)}
                for src in args.images
            ]
            safe = {"url": base_url + "/v1/chat/completions",
                    "note": f"{len(payloads)} 个独立请求（--per-image）", "payloads": payloads}
        else:
            payload = {"model": args.model, "stream": False,
                       "messages": build_messages(prompt, args.images, system, dry=True)}
            safe = {"url": base_url + "/v1/chat/completions",
                    "mode": f"function:{func_key}" if func_key else ("keys" if keys else args.mode),
                    "payload": payload}
        _redact(safe)
        print(json.dumps(safe, ensure_ascii=False, indent=2))
        return

    api_key = load_api_key()
    if not api_key:
        sys.stderr.write(
            "缺少 API Key。请设置环境变量 GRSAI_API_KEY，或在脚本同目录放 config.txt"
            "（内容: GRSAI_API_KEY=你的key）。\n"
        )
        sys.exit(2)

    try:
        if structured and args.per_image and len(args.images) > 1:
            results = []
            for src in args.images:
                msgs = build_messages(prompt, [src], system, dry=False)
                resp = call_relay(base_url, api_key, args.model, msgs, timeout=args.timeout)
                obj = parse_json_object(extract_text(resp))
                if obj is None:
                    obj = {"_raw": extract_text(resp)}
                results.append(obj)
            final = results
        else:
            msgs = build_messages(prompt, args.images, system, dry=False)
            resp = call_relay(base_url, api_key, args.model, msgs, timeout=args.timeout)
            if args.json:
                print(json.dumps(resp, ensure_ascii=False, indent=2))
                return
            if structured:
                obj = parse_json_object(extract_text(resp))
                final = obj if obj is not None else {"_raw": extract_text(resp)}
            else:
                final = extract_text(resp)
    except RuntimeError as e:
        sys.stderr.write(f"调用失败: {e}\n")
        sys.exit(1)

    out = json.dumps(final, ensure_ascii=False, indent=2) if isinstance(final, (dict, list)) else str(final)
    if args.output and isinstance(final, (dict, list)):
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        sys.stderr.write(f"[ok] 已写入: {args.output}\n")
    print(out)


def _redact(safe):
    """dry-run 输出中抹掉 data URL 的字节，避免打印超大内容。"""
    node = safe.get("payload")
    nodes = [node] if node else safe.get("payloads", [])
    for p in nodes:
        for m in p.get("messages", []):
            if isinstance(m.get("content"), list):
                for part in m["content"]:
                    if part.get("type") == "image_url":
                        u = part["image_url"]["url"]
                        if u.startswith("data:") and "," in u:
                            head, rest = u.split(",", 1)
                            part["image_url"]["url"] = f"{head},<{len(rest)} bytes base64>"


if __name__ == "__main__":
    main()
