---
name: vision-capability
description: 通用「视觉能力」，可独立抽取、与非视觉模型配合。通过 grsai 中转站调用 Google Gemini 多模态模型，把图片转成纯文本/JSON，为不具备视觉的模型提供"眼睛"。提供 OCR 文字识别、内容描述、结构化提取、面向非视觉模型的详尽「感知(perceive)」报告，以及按任意字段名做领域无关的「keys 提取」。不含任何业务领域内容（如商品/SKU），纯标准库、无状态、凭证不外泄。可被其它技能/非视觉 agent 直接 subprocess 调用，也可作为 MCP 工具（scripts/mcp_server.py）暴露。当用户需要识别图片文字、描述图片内容、从图片/截图/表格提取结构化数据、给下游非视觉模型补齐多模态能力，或希望使用 Gemini 视觉能力（而当前模型不具备视觉）时调用。
---

# vision-capability — 通用视觉能力（服务于非视觉模型）

把图片转成**纯文本/JSON**的视觉转述器。本身不做业务判断，只负责"看见并转述"，
交由下游不会看图的模型去推理、决策、生成。与 `grsai-vision`（含 SKU 领域功能）不同，
本技能**完全不含业务领域内容**，是纯粹的、可复用的视觉原语。

## 何时使用
- 需要**识别图片文字（OCR）**。
- 需要**描述图片内容**或让非视觉模型"理解"一张图。
- 需要从图片**提取结构化数据 / 表格**。
- 需要**按任意字段名**做领域无关提取（如 `名称,材质,价格`），无需预定义类目。
- 给下游**非视觉模型**补齐多模态能力（通过 `--mode perceive` 产出详尽感知报告）。
- 用户明确要求用 **Gemini 视觉** 能力，或通过 **grsai 中转站** 处理图片。

## 配置（必须）
只需一个凭证，从**环境变量**或**本地 config 文件**读取（不进对话、不写日志）：

```bash
# 方式 A：环境变量（推荐，临时）
export GRSAI_API_KEY="你的grsai中继Key"
# 可选：自定义中转站地址，默认 https://grsaiapi.com
export GRSAI_CHAT_BASE_URL="https://grsaiapi.com"
```

```txt
# 方式 B：脚本同目录 config.txt（内容一行即可，勿提交仓库）
GRSAI_API_KEY=你的grsai中继Key
```

Key 在 GRSAI 控制台获取（测试页 Global Configuration 里的 API Key 字段）。

## 调用方式
脚本：`scripts/vision.py`（仅用 Python 标准库，无需安装依赖）。

```bash
# ★★★ 视觉理解智能体（默认 / 唯一模式）：先分类图片类型，再按类型走专项策略，
#     输出固定 JSON Schema（image_type/layout/elements/text_content/style_analysis/
#     spatial_relationships/reconstruction_hint/confidence），供下游非视觉模型做
#     图片理解 / UI复刻 / HTML生成 / 设计分析。
python scripts/vision.py 图片.png

# 多张图片一起识别
python scripts/vision.py a.png b.jpg "https://example.com/c.png"

# 直接把结果写成 JSON 文件
python scripts/vision.py 图片.png --output result.json

# ★ 领域无关提取：传任意字段名，返回 JSON（无需预定义类目）
python scripts/vision.py 产品.png --keys 名称,材质,价格,规格

# 结果直接写 JSON 文件
python scripts/vision.py 产品.png --keys 名称,价格 --output result.json

# 自定义提示词 + 指定模型
python scripts/vision.py 图.png --prompt "提取所有价格与型号" --model gemini-3-pro

# 批量：每张图单独提取，结果聚合为 JSON 数组
python scripts/vision.py 1.png 2.png 3.png --keys 名称 --per-image

# 原始 JSON 响应 / 仅预览请求（无需 Key）
python scripts/vision.py 图.png --json
python scripts/vision.py 产品.png --keys 名称 --dry-run
```

### 参数速查
| 参数 | 说明 |
|------|------|
| 无参数（默认） | 视觉理解智能体：输出固定 JSON Schema（image_type/layout/elements/text_content/style_analysis/spatial_relationships/reconstruction_hint/confidence）|
| `--keys` | 领域无关提取：逗号分隔的字段名，如 `--keys 名称,材质,价格`（返回 JSON）|
| `--function` | 调用 functions.json 里自定义的功能（需自建）|
| `--prompt` | 覆盖默认提示词 |
| `--output` | 将 JSON 结果写入文件 |
| `--per-image` | 多图时每张单独请求，结果聚合为 JSON 数组 |
| `--json` | 输出原始 API 响应 |
| `--dry-run` | 只构造请求 payload，不发起网络请求 |

> 不确定信息会在 JSON 中体现；图片没有的字段填 `null`（不虚构）。

## 作为 MCP 工具使用（可选）
`scripts/mcp_server.py` 是一个最小可用的 MCP stdio 服务，把本能力暴露成工具
`vision_analyze`，任何支持 MCP 的 agent 都能直接调用（无需自己处理凭证/base64/请求）。

在 MCP 客户端配置里指向它，例如：
```json
{
  "mcpServers": {
    "vision-capability": {
      "command": "python",
      "args": ["<本技能目录>/scripts/mcp_server.py"]
    }
  }
}
```
`vision_analyze` 入参：`image`(必填), `mode`(默认 perceive), `keys`, `prompt`, `model`, `hint`。

## 与非视觉模型配合（子进程调用）
非视觉模型"要看图"时，直接调用本脚本，把 stdout 作为图片文字转述拼回上下文：

```python
import subprocess

def see(image_path, mode="perceive"):
    r = subprocess.run(
        ["python", "scripts/vision.py", image_path, "--mode", mode],
        cwd="<本技能目录>", capture_output=True, text=True, encoding="utf-8",
    )
    return r.stdout.strip()   # 纯文本；用 --keys/--mode extract 则为 JSON 文本

desc = see("截图.png")        # perceive 详尽报告
# 把 desc 交给非视觉模型继续分析 / 决策 / 生成
```

## 自定义功能（functions.json，可选）
若某领域有**固定字段+类目**要反复提取，可在脚本同目录建 `functions.json`
（结构见 `grsai-vision` 技能的示例），用 `--function N` 调用。本技能默认**不携带任何领域功能**，
保持纯粹通用。

## 安全说明
- 仅向 `GRSAI_CHAT_BASE_URL`（默认官方中转 `grsaiapi.com`）发送请求，无其它外传地址。
- 凭证只从环境变量 / 本地 `config.txt` 读取，不外泄、不落盘、不打日志。
- 图片在本地编码为 base64 后随请求发出（OpenAI 视觉消息格式）；远程 URL 直接透传。

### 分发前必读（防 Key 泄露）
本技能代码**不含任何密钥**。真实 Key 只可能在本地 `config.txt`（或环境变量）里。
- **分发/提交时请务必排除 `config.txt`**：本目录已提供 `.gitignore` 忽略 `config.txt` 与 `.grsai_key`，并只随附占位模板 `config.example.txt`。
- 不要把含真实 `config.txt` 的旧技能目录（如 `grsai-vision/scripts/config.txt`）整包发出。
- 接收方应使用**他们自己的** grsai Key（环境变量或自建 `config.txt`），与你的账号无关。
- 工具输出永不打印/记录 Authorization 头；`--dry-run` 还会主动抹掉图片 base64 字节。
