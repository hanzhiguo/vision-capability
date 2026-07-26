# vision-capability

通用「视觉能力」——通过 [grsai](https://grsaiapi.com) 中转站调用 Google Gemini 多模态模型，
把**图片转成纯文本 / JSON**，为**不具备视觉的模型**提供"眼睛"。

> 本技能**不含任何业务领域内容**（如商品 / SKU），是一个纯粹的、可复用的视觉原语。
> 代码零硬编码密钥，凭证只从环境变量或本地 `config.txt` 读取。

## 它解决什么

很多语言模型不会"看图"。本技能把图片编码后发给 Gemini，只回吐**纯文本 / 结构化 JSON**，
因此任何只会读文字的模型、agent、工作流，都能借它"看见"并继续推理、决策、生成。

## 功能

| 模式 | 输出 | 适用场景 |
|------|------|----------|
| `ocr` | 图中全部文字（表格保留行列） | 需要精确引用图中文字 |
| `describe` | 一句话 / 段落描述 | 一句话画像 |
| `extract` | JSON（自动推断字段） | 要结构化但字段不固定 |
| `perceive` | 详尽 Markdown 报告（概述 / OCR / 对象 / 布局 / 颜色 / 数据 / 需核验） | **非视觉模型理解整图并推理（推荐）** |
| `--keys 名称,材质,价格` | JSON（指定字段，领域无关） | 按需取任意字段，无需预定义类目 |

所有不确定信息都标注 `（核验）`；图片没有的字段填 `null`（不虚构）。

## 快速开始

### 1. 配置 grsai Key

Key 来自你的 grsai 中转站账号（grsaiapi.com 控制台 → Global Configuration → API Key）。二选一：

```bash
# 方式 A：环境变量（推荐）
export GRSAI_API_KEY="你的grsai中继Key"

# 方式 B：本地文件（复制模板后填入）
cp config.example.txt config.txt
# 然后编辑 config.txt，把 GRSAI_API_KEY=你的grsai中继Key 填上
```

### 2. 使用

脚本仅依赖 Python 标准库，无需 `pip install`。

```bash
# 详尽感知报告（面向非视觉模型）
python scripts/vision.py 图片.png --mode perceive

# OCR 识别文字
python scripts/vision.py 图片.png

# 多张图
python scripts/vision.py a.png b.jpg "https://example.com/c.png"

# 按任意字段名做领域无关提取
python scripts/vision.py 产品.png --keys 名称,材质,价格

# 结果写 JSON 文件 / 指定模型
python scripts/vision.py 产品.png --keys 名称,价格 --output result.json --model gemini-3-pro

# 批量：每张图单独提取，结果聚合为 JSON 数组
python scripts/vision.py 1.png 2.png 3.png --keys 名称 --per-image

# 只看请求、不真正调用（无需 Key）
python scripts/vision.py 图片.png --mode perceive --dry-run
```

## 作为 MCP 工具使用（可选）

`scripts/mcp_server.py` 是一个最小可用的 MCP stdio 服务，把本能力暴露成工具
`vision_analyze`，任何支持 MCP 的 agent 都能直接调用（无需自己处理凭证 / base64 / 请求构造）。

在 MCP 客户端配置里指向它：

```json
{
  "mcpServers": {
    "vision-capability": {
      "command": "python",
      "args": ["<本仓库目录>/scripts/mcp_server.py"]
    }
  }
}
```

`vision_analyze` 入参：`image`(必填), `mode`(默认 `perceive`), `keys`, `prompt`, `model`, `hint`。

## 与非视觉模型配合（子进程调用）

非视觉模型"要看图"时，直接调用本脚本，把 stdout 作为图片文字转述拼回上下文：

```python
import subprocess

def see(image_path, mode="perceive"):
    r = subprocess.run(
        ["python", "scripts/vision.py", image_path, "--mode", mode],
        cwd="<本仓库目录>", capture_output=True, text=True, encoding="utf-8",
    )
    return r.stdout.strip()   # 纯文本；用 --keys/--mode extract 则为 JSON 文本

desc = see("截图.png")        # perceive 详尽报告
# 把 desc 交给非视觉模型继续分析 / 决策 / 生成
```

## 自定义功能（functions.json，可选）

若某领域有**固定字段 + 类目**要反复提取，可在 `scripts/` 目录建 `functions.json`
（结构参考其他技能示例），用 `--function N` 调用。本仓库默认**不携带任何领域功能**，保持纯粹通用。

## 安全

- 仅向 `GRSAI_CHAT_BASE_URL`（默认官方中转 `grsaiapi.com`）发送请求，无其它外传地址。
- 凭证只从环境变量 / 本地 `config.txt` 读取，不外泄、不落盘、不打日志；工具输出永不打印 Authorization 头。
- **分发 / 提交时请务必排除 `config.txt`**（本仓库 `.gitignore` 已忽略它）。接收方应使用他们自己的 grsai Key。

## 许可证

[MIT](./LICENSE)
