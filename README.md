# CopyMyVoice — 中文 → 你的音色说英语

把你的一段中文讲话变成 **你本人音色** 的英语讲话，并保留你的 **语速**。

```
中文音频/麦克风 → 中文识别(Whisper) → 中→英翻译(LLM) → 你的音色说英语(F5-TTS) → 语速对齐 → 输出 wav
```

全程本地运行，免费、隐私好。需要一块 NVIDIA 显卡（≥ 4 GB 显存）。

两个使用入口：

- **CLI**：`python clone.py --input my_chinese.wav`
- **Web UI**（推荐）：双击 `copymyvoice.bat`，浏览器自动打开 `http://127.0.0.1:7860`

---

## 1. 系统要求

| 项目 | 要求 |
|---|---|
| OS | Windows 10 / 11（macOS / Linux 也可） |
| Python | 3.10 / 3.11 / 3.12 |
| GPU | NVIDIA，≥ 4 GB 显存（已装好驱动） |
| 磁盘 | ≥ 8 GB（存模型） |
| 麦克风 | 仅文件模式不需要 |

> 没 GPU 也能跑，CPU 模式慢约 10–30×。

---

## 2. 安装

### 2.1 创建虚拟环境

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2.2 装 PyTorch (CUDA 版)

去 [pytorch.org/get-started](https://pytorch.org/get-started/locally/) 选你的 CUDA 版本，例如 CUDA 12.4：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

测一下：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

应输出 `True`。

### 2.3 装其余依赖

```powershell
pip install -r requirements.txt
```

`f5-tts` 1.1.x 在 Windows 上首次加载会读 `cudnn64_9.dll`，脚本已自动处理 PATH 一般没事。

### 2.4 (可选) 装 ollama 跑翻译

默认用离线的 `argostranslate`，效果一般。装本地 LLM 翻译质量会好很多：

```powershell
# https://ollama.com/download
ollama pull qwen2.5:7b
ollama serve
```

`http://localhost:11434` 跑着即可，脚本自动检测。

### 2.5 (可选) 装 rubberband

```powershell
scoop install rubberband
```

没装也能跑，librosa 兜底。

---

## 3. 使用

### 3.1 Web UI（推荐）

双击 `copymyvoice.bat`，浏览器自动开 http://127.0.0.1:7860。

- 上传 wav / m4a / mp3 文件，**或** 按"录音"按钮录中文
- 首次跑完后，结果区会显示：中文识别（可编辑）、英文翻译（可编辑）、输出音频
- **两步校验**：
  1. 改完中文 → 点「🌐 翻译」，看英文翻译对不对
  2. 英文不对就改 → 点「🎙️ 生成语音」（跳过翻译直接合成）
- 等 10–60 秒（看音频长度），输出 wav 自动保存在 `output/`

### 3.2 命令行

```powershell
# 录音
python clone.py

# 用文件
python clone.py --input my_chinese.wav

# 指定输出
python clone.py --input my_chinese.wav --out result.wav
```

---

## 4. 流程细节

| 步骤 | 模型 | 作用 |
|---|---|---|
| ASR | faster-whisper `large-v3` | 中文 → 文字 + 时长 |
| 翻译 | ollama `qwen2.5:7b`（首选）或 argostranslate | 中文 → 英文 |
| TTS | F5-TTS `F5TTS_v1_Base` | 跨语言零样本音色克隆 |
| 语速对齐 | pyrubberband / librosa | 把英文长度对齐到中文时长 |

**怎么保留语速**：你说了 8 秒中文 → 英文先按自然节奏生成（可能 5s）→ 整段拉到 8s。这样语速比例一致，但听起来还是流畅的英语，不是逐字蹦。

**音色克隆原理**：用音频前 6 秒 + 它的中文字幕作为 voice prompt。F5-TTS 把目标文本（英文）当成下一句接在你说的那句后面，自然继承音色。

---

## 5. 常见问题

**Q: 报 `Could not load symbol cudnnGetLibConfig. Error code 127`？**
A: `steps.py` 顶部的 `_preload_cudnn()` 已经处理过 PATH；如果还报，确认下 PyTorch 是 CUDA 版且 ≥ 2.6。

**Q: 输出有中英混杂？**
A: 这是 F5-TTS 跨语言+短 ref_text 的已知问题。代码里通过 `fix_duration` 参数强制单次生成（不切段）解决。F5-TTS 单次最长 30s，输入超 30s 需要先拆分。

**Q: 输出英文有中国口音 / 中文音素漏出？**
A: 默认 `quality_mode=True` 会把 ref_text（中文）翻译成英文后再喂给 F5-TTS，避免中文 pinyin 渗到英文输出。如果想用纯中文 ref_text，在 UI 关掉 quality_mode（API 加 `quality_mode=false`）。

**Q: 输出英文有奇怪的信噪比 / 跟原声不像？**
A: 录 10–30 秒安静、自然语速的中文给 F5-TTS 作 voice prompt。太短 / 太嘈杂 / 太夸张的语气都会失真。`quality_mode=True` 还会做 preemphasis 高通滤波 + 峰值归一化，让声音更清晰。

**Q: 想换翻译模型？**
A: 改 `steps.py` 里 `_translate_ollama` 的 `model` 列表，比如换成 `qwen2.5:14b` 或 `llama3.1:8b`。

**Q: 输出 wav 转 mp3？**
A: `ffmpeg -i output/result.wav result.mp3`。

---

## 6. 文件结构

```
copymyvoice/
├── clone.py             # CLI 入口
├── app.py               # Gradio UI（已弃用，保留作 fallback）
├── web_server.py        # 当前用的 Web UI 后端（Python stdlib + 自定义 HTML）
├── web/
│   └── index.html       # 前端页面
├── pipeline.py          # 流程编排
├── steps.py             # 各步骤实现
├── requirements.txt
├── copymyvoice.bat      # Windows 一键启动
└── README.md
```

模型默认下载到 `~/.cache/huggingface`，可改 `HF_HOME` 环境变量。

---

## 7. 路线图

- [ ] 实时流式（边录边处理）
- [ ] 多说话人切换
- [ ] 用 CosyVoice / XTTS 替代 F5-TTS
- [ ] 端到端 LLM（一次调用 = ASR + 翻译 + 音色 prompt）
