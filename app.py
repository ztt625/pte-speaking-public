#!/usr/bin/env python3
"""
🦊小狐泥PTE口语诊断 — 公开版
部署于 Hugging Face Spaces
"""

import sys
import os
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nltk
try:
    nltk.download("cmudict", quiet=True)
except Exception:
    pass

import gradio as gr

from analyze_v2 import run_analysis, _get_whisper_model
from export_html import build_single_student_html

CUSTOM_CSS = """
.gradio-container { max-width: 640px !important; margin: 0 auto !important; min-height: 100vh; }
body, .gradio-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif !important; }
footer { display: none !important; }
/* 隐藏 loading 条，避免手机页面抖动 */
#loading, .progress-bar, .eta-bar, .wrap.hide, .generating, .pending { display: none !important; }
.wrap.default { min-height: auto !important; }
/* 底部进度条 */
.interface-queue, .dark .interface-queue { display: none !important; }
"""


def diagnose(audio_file, text):
    """上传录音 → 分析 → 导出学生版 HTML 报告"""
    if audio_file is None:
        return None, "### ⚠️ 请先上传录音再点诊断"

    # Gradio 6 兼容：audio 可能返回 tuple (sr, array)
    audio_path = audio_file
    tmp_file = None
    if isinstance(audio_file, tuple):
        try:
            import soundfile as sf
            sr, audio_array = audio_file
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp_file.name, audio_array, sr)
            audio_path = tmp_file.name
        except Exception:
            pass

    text_clean = text.strip() if text else None

    try:
        result = run_analysis(
            audio_path=audio_path,
            ref_path=None,
            text=text_clean,
            model_name="tiny",
            student_name="同学",
            question_type="RA",
        )
    except Exception as e:
        tb = traceback.format_exc()
        # 出错时也要清理临时文件
        if tmp_file:
            try:
                os.unlink(tmp_file.name)
            except Exception:
                pass
        return None, f"### ❌ 分析失败\n\n{str(e)}\n\n<details><summary>详情</summary>\n\n```\n{tb}\n```\n</details>"

    # 导出学生版 HTML（必须在清理 tmp 文件之前，波形图需要音频文件）
    export_data = {
        "student_name": "同学",
        "audio_path": audio_path if (audio_path and os.path.exists(audio_path)) else "",
        "result": result,
        "markdown": result["markdown"],
        "question_type": "RA",
        "text": text_clean or "",
    }

    try:
        html_path = build_single_student_html(export_data)
        summary = (
            f"### ✅ 诊断完成\n\n"
            f"| 指标 | 数值 |\n"
            f"|------|------|\n"
            f"| 🗣 语速 | {result['rhythm']['wpm']} WPM |\n"
            f"| 📏 蹦词指数 | {result['rhythm']['staccato_score']}/6 |\n"
            f"| ⚖️ 虚实比 | {result['rhythm']['content_func_ratio']}:1 |\n\n"
            f"📄 报告已生成 👆 **点击上方文件名称即可下载**（下载后用浏览器打开查看交互式诊断）"
        )
        return html_path, summary
    except Exception as e:
        return None, f"### ⚠️ 报告生成失败\n\n{str(e)}"
    finally:
        # 导出完成后再清理临时文件
        if tmp_file:
            try:
                os.unlink(tmp_file.name)
            except Exception:
                pass


with gr.Blocks(
    title="🦊小狐泥PTE口语诊断",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(),
) as demo:

    gr.Markdown(
        """
        # 🦊 小狐泥 PTE 口语诊断 ⚡公测版

        免费 AI 口语诊断工具。上传你的 PTE 口语录音，
        一键生成交互式诊断报告。看到自己的问题，才能针对性提分。

        > ⏱️ 每次诊断约需 **1-2 分钟**（免费服务器处理中，请耐心等待）

        """
    )

    audio_input = gr.Audio(label="🎙️ 上传录音（最长 2 分钟）")
    text_input = gr.Textbox(label="📖 原文（可选，填写后自动对比发音偏差）", placeholder="粘贴原文，诊断更精准...", lines=2)
    analyze_btn = gr.Button("🔍 开始诊断", variant="primary", size="lg")

    report_html = gr.File(label="📄 诊断报告 — 点击文件名下载", visible=True)
    status_md = gr.Markdown("上传录音后点击「开始诊断」")

    analyze_btn.click(
        fn=diagnose,
        inputs=[audio_input, text_input],
        outputs=[report_html, status_md],
    )

    gr.Markdown(
        """
        ---
        > ⚡ 当前为免费公测版，基于 AI 模型自动分析，结果仅供参考。
        """
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
