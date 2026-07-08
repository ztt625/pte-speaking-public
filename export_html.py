#!/usr/bin/env python3
"""PTE 口语诊断 — 学生版网页导出
================================
单题诊断和批量诊断的学生版自包含 HTML 导出。
"""

import os
import sys
import re
from pathlib import Path

# Ensure the scripts dir is importable
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime
import markdown as md_lib

# 从同目录的 ui_components 导入波形图生成函数
from ui_components import build_interactive_waveform_html


def build_single_student_html(export_data):
    """将单题诊断结果导出为自包含的学生练习网页。

    export_data: {student_name, audio_path, result, markdown, question_type, text}
    返回文件路径，失败返回 None。
    """
    if not export_data:
        return None

    student_name = export_data.get("student_name", "学生")
    audio_path = export_data.get("audio_path", "")
    result = export_data.get("result", {})
    markdown = export_data.get("markdown", "")
    qtype = export_data.get("question_type", "RA")
    text = export_data.get("text", "")
    date_str = datetime.now().strftime("%Y-%m-%d")

    # ── 交互式波形 iframe ──
    waveform_iframe = ""
    if audio_path and os.path.exists(audio_path) and result:
        try:
            waveform_iframe = build_interactive_waveform_html(
                audio_path, result, wrap_iframe=True
            )
        except Exception:
            waveform_iframe = ""
    if not waveform_iframe:
        wf_b64 = result.get("waveform_b64", "")
        if wf_b64:
            waveform_iframe = (
                f'<img src="data:image/png;base64,{wf_b64}" '
                f'style="max-width:100%;border-radius:6px;" alt="波形图">'
            )
        else:
            waveform_iframe = (
                '<p style="color:#999;font-style:italic;">⚠️ 波形图不可用</p>'
            )

    # ── 清理诊断 Markdown（去掉波形章节，外面已展示）──
    diagnosis_md_clean = markdown
    diagnosis_md_clean = re.sub(
        r'\n?## 📈 诊断波形.*?(?=\r?\n## |\Z)', '', diagnosis_md_clean, flags=re.DOTALL
    )
    diagnosis_md_clean = re.sub(
        r'<div class="iwf-root".*?</div>\s*\n?', '', diagnosis_md_clean, flags=re.DOTALL
    )
    diagnosis_md_clean = re.sub(
        r'<audio class="iwf-audio"[^>]*>.*?</audio>\s*\n?', '', diagnosis_md_clean, flags=re.DOTALL
    )
    diagnosis_md_clean = re.sub(
        r'<img[^>]*alt="诊断波形图"[^>]*/?>\s*\n?', '', diagnosis_md_clean
    )
    diagnosis_md_clean = re.sub(r'\n{3,}', '\n\n', diagnosis_md_clean)

    diagnosis_html = md_lib.markdown(
        diagnosis_md_clean,
        extensions=["tables", "fenced_code", "codehilite", "nl2br"],
    )

    # ── 主要问题摘要 ──
    flags = result.get("flags", {})
    problem_lines = []
    for dim, items_list in flags.items():
        for item in items_list:
            problem_lines.append(f"<li>{item}</li>")
    problems_html = (
        f'<ul class="problem-list">{"".join(problem_lines[:8])}</ul>'
        if problem_lines
        else ""
    )

    # ── 指标徽章 ──
    rhythm = result.get("rhythm", {})
    wpm = rhythm.get("wpm", "—")
    staccato = rhythm.get("staccato_score", "—")
    ratio = rhythm.get("content_func_ratio", "—")

    # ── 拼装 HTML 页面 ──
    html = _STUDENT_PAGE_TEMPLATE.format(
        student_name=student_name,
        date_str=date_str,
        qtype=qtype,
        wpm=wpm,
        staccato=staccato,
        ratio=ratio,
        text_section=f'<p class="original-text"><strong>📖 原文：</strong>{text}</p>' if text else '',
        waveform_iframe=waveform_iframe,
        problems_section=f'<div class="problems"><strong>⚠️ 主要问题：</strong>{problems_html}</div>' if problems_html else '',
        diagnosis_html=diagnosis_html,
    )

    export_dir = Path("/tmp/exported_pages")
    export_dir.mkdir(parents=True, exist_ok=True)
    safe_name = student_name.replace("/", "_").replace(" ", "_") if student_name else "学生"
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = f"{safe_name}_{qtype}_{ts}.html"
    filepath = export_dir / filename
    filepath.write_text(html, encoding="utf-8")
    return str(filepath)


def build_batch_student_export_html(batch_data):
    """将批量诊断结果导出为自包含的学生练习网页。

    网页特性：
    - 所有音频内嵌为 base64，离线可用
    - 每道题配有交互式波形图（可点击跳转、播放同步）
    - 综合诊断报告（学生友好版）
    - 白底灰背景卡片式布局，适合打印
    返回文件路径，失败返回 None。
    """
    if not batch_data or not batch_data.get("questions"):
        return None

    student_name = batch_data.get("student_name", "学生")
    synthesis_md = batch_data.get("synthesis_md", "")
    questions = batch_data.get("questions", [])
    date_str = datetime.now().strftime("%Y-%m-%d")

    # ── 综合报告 Markdown → HTML ──
    synthesis_html = md_lib.markdown(
        synthesis_md,
        extensions=["tables", "fenced_code", "codehilite", "nl2br"],
    )

    # ── 逐题卡片 ──
    question_cards_html = []
    for q in questions:
        qtype = q.get("type", "RA")
        idx = q.get("idx", 1)
        audio_path = q.get("audio_path", "")
        result = q.get("result", {})
        diagnosis_md = q.get("diagnosis_md", "")
        text = q.get("text", "")

        # 生成自包含的交互式波形 iframe（音频 + 波形 + 播放控件一体化）
        waveform_iframe = ""
        if audio_path and os.path.exists(audio_path) and result:
            try:
                waveform_iframe = build_interactive_waveform_html(
                    audio_path, result, wrap_iframe=True
                )
            except Exception:
                waveform_iframe = ""
        if not waveform_iframe:
            wf_b64 = result.get("waveform_b64", "") if result else ""
            if wf_b64:
                waveform_iframe = (
                    f'<img src="data:image/png;base64,{wf_b64}" '
                    f'style="max-width:100%;border-radius:6px;" alt="波形图">'
                )
            else:
                waveform_iframe = (
                    '<p style="color:#999;font-style:italic;">⚠️ 波形图不可用</p>'
                )

        # 诊断内容 Markdown → HTML（先清理整个波形图章节，外面已展示）
        diagnosis_md_clean = diagnosis_md
        diagnosis_md_clean = re.sub(
            r'\n?## 📈 诊断波形.*?(?=\r?\n## |\Z)', '', diagnosis_md_clean, flags=re.DOTALL
        )
        diagnosis_md_clean = re.sub(
            r'<div class="iwf-root".*?</div>\s*\n?', '', diagnosis_md_clean, flags=re.DOTALL
        )
        diagnosis_md_clean = re.sub(
            r'<audio class="iwf-audio"[^>]*>.*?</audio>\s*\n?', '', diagnosis_md_clean, flags=re.DOTALL
        )
        diagnosis_md_clean = re.sub(
            r'<img[^>]*alt="诊断波形图"[^>]*/?>\s*\n?', '', diagnosis_md_clean
        )
        diagnosis_md_clean = re.sub(r'\n{3,}', '\n\n', diagnosis_md_clean)

        diagnosis_html = md_lib.markdown(
            diagnosis_md_clean,
            extensions=["tables", "fenced_code", "codehilite", "nl2br"],
        )

        # 提取主要问题摘要
        flags = result.get("flags", {}) if result else {}
        problem_lines = []
        for dim, items_list in flags.items():
            for item in items_list:
                problem_lines.append(f"<li>{item}</li>")
        problems_html = (
            f'<ul class="problem-list">{"".join(problem_lines[:6])}</ul>'
            if problem_lines
            else ""
        )

        # 指标徽章
        rhythm = result.get("rhythm", {}) if result else {}
        wpm = rhythm.get("wpm", "—")
        staccato = rhythm.get("staccato_score", "—")
        ratio = rhythm.get("content_func_ratio", "—")

        question_cards_html.append(f'''
        <div class="question-card">
            <div class="card-header">
                <span class="badge">{qtype} #{idx}</span>
                <span class="metrics">
                    <span title="语速">🗣 {wpm} WPM</span>
                    <span title="蹦词指数">📏 {staccato}/6</span>
                    <span title="虚实比">⚖️ {ratio}:1</span>
                </span>
            </div>
            {f'<p class="original-text"><strong>📖 原文：</strong>{text}</p>' if text else ''}
            <div class="waveform-wrap">
                {waveform_iframe}
            </div>
            {f'<div class="problems"><strong>⚠️ 主要问题：</strong>{problems_html}</div>' if problems_html else ''}
            <details class="diagnosis-details">
                <summary>📋 完整诊断详情</summary>
                <div class="diagnosis-content">
                    {diagnosis_html}
                </div>
            </details>
        </div>
        ''')

    all_cards = "\n".join(question_cards_html)

    html = _BATCH_PAGE_TEMPLATE.format(
        student_name=student_name,
        date_str=date_str,
        synthesis_html=synthesis_html,
        all_cards=all_cards,
    )

    export_dir = Path("/tmp/exported_pages")
    export_dir.mkdir(parents=True, exist_ok=True)
    safe_name = student_name.replace("/", "_").replace(" ", "_") if student_name else "学生"
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = f"{safe_name}_批量诊断_{ts}.html"
    filepath = export_dir / filename
    filepath.write_text(html, encoding="utf-8")
    return str(filepath)


# ══════════════════════════════════════════════════
# HTML 模板
# ══════════════════════════════════════════════════

_STUDENT_PAGE_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{student_name} — PTE 口语诊断报告</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "PingFang SC", sans-serif;
    max-width: 960px; margin: 0 auto; padding: 20px 20px 60px;
    background: #f5f5f5; color: #2c2c2c; line-height: 1.75; font-size: 15px;
  }}
  header {{
    text-align: center; padding: 32px 20px 20px;
    background: linear-gradient(135deg, #e67e22 0%, #d35400 100%);
    color: #fff; border-radius: 12px; margin-bottom: 28px;
  }}
  header h1 {{ font-size: 1.6em; margin: 0 0 6px; font-weight: 700; }}
  header .date {{ font-size: 0.85em; opacity: 0.85; margin: 0; }}
  h2 {{
    font-size: 1.2em; margin: 28px 0 14px; padding-bottom: 6px;
    border-bottom: 2px solid #e67e22; color: #333;
  }}
  h3 {{ font-size: 1.05em; margin: 16px 0 8px; color: #555; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #ddd; padding: 5px 10px; text-align: left; }}
  th {{ background: #f0f0f0; font-weight: 600; }}
  img {{ max-width: 100%; height: auto; border-radius: 6px; }}
  blockquote {{
    border-left: 3px solid #e67e22; margin: 10px 0; padding: 6px 14px;
    background: #fff; border-radius: 0 6px 6px 0; color: #555;
  }}
  code {{ background: #eee; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 20px 0; }}

  .question-card {{
    background: #fff; border-radius: 10px; padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}

  .card-header {{
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 8px; margin-bottom: 12px;
  }}
  .badge {{
    display: inline-block; background: #e67e22; color: #fff;
    padding: 3px 14px; border-radius: 20px; font-size: 14px; font-weight: 700;
  }}
  .metrics {{
    display: flex; gap: 12px; font-size: 13px; color: #777;
    background: #f9f9f9; padding: 4px 12px; border-radius: 16px;
  }}
  .metrics span {{ white-space: nowrap; }}

  .original-text {{
    background: #fff8e1; padding: 10px 14px; border-radius: 6px;
    font-size: 15px; line-height: 1.6; color: #5d4037;
    border-left: 3px solid #ffc107; margin: 10px 0;
  }}

  .waveform-wrap {{ margin: 14px 0; }}
  .waveform-wrap iframe {{
    width: 100%; height: 255px; border: none; border-radius: 8px;
  }}

  .problems {{
    background: #fff5f5; padding: 10px 14px; border-radius: 6px;
    font-size: 13px; color: #7b3f3f; margin: 10px 0;
    border-left: 3px solid #e74c3c;
  }}
  .problem-list {{ margin: 4px 0 0; padding-left: 18px; }}
  .problem-list li {{ margin: 2px 0; }}

  .diagnosis-details {{
    margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px;
  }}
  .diagnosis-details summary {{
    cursor: pointer; color: #e67e22; font-weight: 600; font-size: 14px;
    user-select: none;
  }}
  .diagnosis-details summary:hover {{ color: #d35400; }}
  .diagnosis-content {{ margin-top: 10px; font-size: 14px; }}
  .diagnosis-content h2 {{ font-size: 1.1em; }}
  .diagnosis-content h3 {{ font-size: 1em; }}

  footer {{
    text-align: center; margin-top: 40px; padding: 16px;
    color: #999; font-size: 13px; border-top: 1px solid #e0e0e0;
  }}

  .teacher-only {{ display: none !important; }}

  .practice-guide {{
    background: linear-gradient(135deg, #e8f4fd 0%, #f0f7fc 100%);
    border-radius: 10px; padding: 20px 24px;
    margin: 24px 0;
    border-left: 4px solid #3498db;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  }}
  .practice-guide h2 {{
    border-bottom: none !important; margin-top: 0 !important;
    color: #2c3e50; font-size: 1.15em;
  }}
  .practice-guide h3 {{
    color: #2980b9; margin: 16px 0 6px; font-size: 1em;
  }}
  .practice-guide ol, .practice-guide ul {{ margin: 6px 0; padding-left: 20px; }}
  .practice-guide li {{ margin: 4px 0; }}
  .practice-guide blockquote {{
    border-left-color: #3498db; background: #fff;
    margin: 8px 0; padding: 8px 14px;
  }}

  @media print {{
    body {{ background: #fff; font-size: 12px; }}
    .question-card {{ box-shadow: none; border: 1px solid #ddd; }}
    .practice-guide {{ box-shadow: none; border: 1px solid #3498db; }}
    header {{ background: #e67e22 !important; -webkit-print-color-adjust: exact; }}
    .waveform-wrap iframe {{ height: 200px; }}
    @page {{ margin: 12mm; }}
  }}

  @media (max-width: 640px) {{
    body {{ padding: 10px; font-size: 14px; }}
    .card-header {{ flex-direction: column; align-items: flex-start; }}
    header {{ padding: 20px 12px; }}
    header h1 {{ font-size: 1.3em; }}
  }}
</style>
</head>
<body>

<header>
  <h1>🎯 {student_name} — PTE 口语诊断报告</h1>
  <p class="date">生成日期：{date_str} · 小狐泥 PTE 口语诊断</p>
</header>

<section class="question-card">
  <div class="card-header">
    <span class="badge">{qtype}</span>
    <span class="metrics">
      <span title="语速">🗣 {wpm} WPM</span>
      <span title="蹦词指数">📏 {staccato}/6</span>
      <span title="虚实比">⚖️ {ratio}:1</span>
    </span>
  </div>
  {text_section}
  <div class="waveform-wrap">
    {waveform_iframe}
  </div>
  {problems_section}
  <details class="diagnosis-details">
    <summary>📋 完整诊断详情</summary>
    <div class="diagnosis-content">
      {diagnosis_html}
    </div>
  </details>
</section>

<section class="practice-guide">
  <h2>📖 独立练习指引</h2>

  <h3>第一步：先看懂你的诊断</h3>
  <p>点击上方的 <strong>「📋 完整诊断详情」</strong> 展开 AI 详细点评。重点看两样东西：</p>
  <ul>
    <li>🔴 <strong>主要问题</strong> — 这是你当前最扣分的地方</li>
    <li>🟡 <strong>为什么</strong> — 不是"读错了"，而是"为什么会读成这样"，理解根因才能改</li>
  </ul>

  <h3>第二步：跟着指导练感觉</h3>
  <p>诊断里会给出具体的朗读建议，练习时重点关注两个感觉：</p>
  <blockquote>
    <p>🗣 <strong>重读感</strong> — 把有实际意思的词（名词、动词、形容词）读得稍重、稍长，语法功能词（a, the, of, in, and）轻轻带过。找"一重一轻、一重一轻"的节奏，就像波浪一样有起伏。</p>
  </blockquote>
  <blockquote>
    <p>🌊 <strong>连贯感</strong> — 词和词之间不要断开，像水流一样自然地连过去。尤其注意辅音结尾 + 元音开头的连读（比如 "read‿it"、"far‿away"），不要一个字一个字往外蹦。</p>
  </blockquote>

  <h3>第三步：用音频做模仿（最有效！）</h3>
  <ol>
    <li><strong>右键点击页面上的音频</strong> → 选择「在新标签页中打开」</li>
    <li>在新标签页的音频播放器右下角，点 <strong>⋮ → 播放速度 → 0.5x</strong></li>
    <li>先用 0.5 倍速<strong>仔细听</strong>自己的录音，对照波形图找问题（哪里断了？哪里蹦了？）</li>
    <li>再切回<strong>正常速度</strong>，跟着录音<strong>逐句跟读模仿</strong>，尽量还原母语者的节奏</li>
    <li>每道题反复练 3-5 遍，直到你的朗读和录音"越来越像"</li>
  </ol>

  <blockquote>
    <p>💡 <strong>小贴士</strong>：模仿时不要求快，先求"像"——节奏像了、连起来了，分数自然就上去了。每天挑 2-3 道题这样练，一周后你会明显感觉到不同。</p>
  </blockquote>
</section>

<footer>
  <p>🦊 小狐泥 PTE 口语诊断 · <strong>加微信 tonigjw</strong> 了解一对一辅导</p>
  <p style="font-size:12px;color:#999;">生成日期：{date_str}</p>
</footer>

</body>
</html>'''


_BATCH_PAGE_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{student_name} — PTE 口语诊断报告</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "PingFang SC", sans-serif;
    max-width: 960px; margin: 0 auto; padding: 20px 20px 60px;
    background: #f5f5f5; color: #2c2c2c; line-height: 1.75; font-size: 15px;
  }}
  header {{
    text-align: center; padding: 32px 20px 20px;
    background: linear-gradient(135deg, #e67e22 0%, #d35400 100%);
    color: #fff; border-radius: 12px; margin-bottom: 28px;
  }}
  header h1 {{ font-size: 1.6em; margin: 0 0 6px; font-weight: 700; }}
  header .date {{ font-size: 0.85em; opacity: 0.85; margin: 0; }}
  h2 {{
    font-size: 1.2em; margin: 28px 0 14px; padding-bottom: 6px;
    border-bottom: 2px solid #e67e22; color: #333;
  }}
  h3 {{ font-size: 1.05em; margin: 16px 0 8px; color: #555; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #ddd; padding: 5px 10px; text-align: left; }}
  th {{ background: #f0f0f0; font-weight: 600; }}
  img {{ max-width: 100%; height: auto; border-radius: 6px; }}
  blockquote {{
    border-left: 3px solid #e67e22; margin: 10px 0; padding: 6px 14px;
    background: #fff; border-radius: 0 6px 6px 0; color: #555;
  }}
  code {{ background: #eee; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 20px 0; }}

  .synthesis {{
    background: #fff; border-radius: 10px; padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06); margin-bottom: 24px;
  }}

  .question-card {{
    background: #fff; border-radius: 10px; padding: 20px 24px;
    margin: 18px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
  }}
  .question-card:hover {{ box-shadow: 0 3px 12px rgba(0,0,0,0.1); }}

  .card-header {{
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 8px; margin-bottom: 12px;
  }}
  .badge {{
    display: inline-block; background: #e67e22; color: #fff;
    padding: 3px 14px; border-radius: 20px; font-size: 14px; font-weight: 700;
  }}
  .metrics {{
    display: flex; gap: 12px; font-size: 13px; color: #777;
    background: #f9f9f9; padding: 4px 12px; border-radius: 16px;
  }}
  .metrics span {{ white-space: nowrap; }}

  .original-text {{
    background: #fff8e1; padding: 10px 14px; border-radius: 6px;
    font-size: 15px; line-height: 1.6; color: #5d4037;
    border-left: 3px solid #ffc107; margin: 10px 0;
  }}

  .waveform-wrap {{ margin: 14px 0; }}
  .waveform-wrap iframe {{
    width: 100%; height: 255px; border: none; border-radius: 8px;
  }}

  .problems {{
    background: #fff5f5; padding: 10px 14px; border-radius: 6px;
    font-size: 13px; color: #7b3f3f; margin: 10px 0;
    border-left: 3px solid #e74c3c;
  }}
  .problem-list {{ margin: 4px 0 0; padding-left: 18px; }}
  .problem-list li {{ margin: 2px 0; }}

  .diagnosis-details {{
    margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px;
  }}
  .diagnosis-details summary {{
    cursor: pointer; color: #e67e22; font-weight: 600; font-size: 14px;
    user-select: none;
  }}
  .diagnosis-details summary:hover {{ color: #d35400; }}
  .diagnosis-content {{ margin-top: 10px; font-size: 14px; }}
  .diagnosis-content h2 {{ font-size: 1.1em; }}
  .diagnosis-content h3 {{ font-size: 1em; }}

  footer {{
    text-align: center; margin-top: 40px; padding: 16px;
    color: #999; font-size: 13px; border-top: 1px solid #e0e0e0;
  }}

  .teacher-only {{ display: none !important; }}

  .practice-guide {{
    background: linear-gradient(135deg, #e8f4fd 0%, #f0f7fc 100%);
    border-radius: 10px; padding: 20px 24px;
    margin: 24px 0;
    border-left: 4px solid #3498db;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  }}
  .practice-guide h2 {{
    border-bottom: none !important; margin-top: 0 !important;
    color: #2c3e50; font-size: 1.15em;
  }}
  .practice-guide h3 {{
    color: #2980b9; margin: 16px 0 6px; font-size: 1em;
  }}
  .practice-guide ol, .practice-guide ul {{ margin: 6px 0; padding-left: 20px; }}
  .practice-guide li {{ margin: 4px 0; }}
  .practice-guide blockquote {{
    border-left-color: #3498db; background: #fff;
    margin: 8px 0; padding: 8px 14px;
  }}

  @media print {{
    body {{ background: #fff; font-size: 12px; }}
    .question-card {{ box-shadow: none; border: 1px solid #ddd; break-inside: avoid; }}
    .synthesis {{ box-shadow: none; border: 1px solid #ddd; }}
    .practice-guide {{ box-shadow: none; border: 1px solid #3498db; }}
    header {{ background: #e67e22 !important; -webkit-print-color-adjust: exact; }}
    .waveform-wrap iframe {{ height: 200px; }}
    @page {{ margin: 12mm; }}
  }}

  @media (max-width: 640px) {{
    body {{ padding: 10px; font-size: 14px; }}
    .card-header {{ flex-direction: column; align-items: flex-start; }}
    header {{ padding: 20px 12px; }}
    header h1 {{ font-size: 1.3em; }}
  }}
</style>
</head>
<body>

<header>
  <h1>🎯 {student_name} — PTE 口语诊断报告</h1>
  <p class="date">生成日期：{date_str} · 小狐泥 PTE 口语诊断</p>
</header>

<section class="synthesis">
  <h2>📝 综合诊断</h2>
  {synthesis_html}
</section>

<section class="practice-guide">
  <h2>📖 独立练习指引</h2>

  <h3>第一步：先看懂你的诊断</h3>
  <p>点击每道题下方的 <strong>「📋 完整诊断详情」</strong> 展开 AI 详细点评。重点看两样东西：</p>
  <ul>
    <li>🔴 <strong>主要问题</strong> — 这是你当前最扣分的地方</li>
    <li>🟡 <strong>为什么</strong> — 不是"读错了"，而是"为什么会读成这样"，理解根因才能改</li>
  </ul>

  <h3>第二步：跟着指导练感觉</h3>
  <p>诊断里会给出具体的朗读建议，练习时重点关注两个感觉：</p>
  <blockquote>
    <p>🗣 <strong>重读感</strong> — 把有实际意思的词（名词、动词、形容词）读得稍重、稍长，语法功能词（a, the, of, in, and）轻轻带过。找"一重一轻、一重一轻"的节奏，就像波浪一样有起伏。</p>
  </blockquote>
  <blockquote>
    <p>🌊 <strong>连贯感</strong> — 词和词之间不要断开，像水流一样自然地连过去。尤其注意辅音结尾 + 元音开头的连读（比如 "read‿it"、"far‿away"），不要一个字一个字往外蹦。</p>
  </blockquote>

  <h3>第三步：用音频做模仿（最有效！）</h3>
  <ol>
    <li><strong>右键点击页面上的音频</strong> → 选择「在新标签页中打开」</li>
    <li>在新标签页的音频播放器右下角，点 <strong>⋮ → 播放速度 → 0.5x</strong></li>
    <li>先用 0.5 倍速<strong>仔细听</strong>自己的录音，对照波形图找问题（哪里断了？哪里蹦了？）</li>
    <li>再切回<strong>正常速度</strong>，跟着录音<strong>逐句跟读模仿</strong>，尽量还原母语者的节奏</li>
    <li>每道题反复练 3-5 遍，直到你的朗读和录音"越来越像"</li>
  </ol>

  <blockquote>
    <p>💡 <strong>小贴士</strong>：模仿时不要求快，先求"像"——节奏像了、连起来了，分数自然就上去了。每天挑 2-3 道题这样练，一周后你会明显感觉到不同。</p>
  </blockquote>
</section>

<section class="questions">
  <h2>🎙️ 各题练习</h2>
  {all_cards}
</section>

<footer>
  <p>🦊 小狐泥 PTE 口语诊断 · <strong>加微信 tonigjw</strong> 了解一对一辅导</p>
  <p style="font-size:12px;color:#999;">生成日期：{date_str}</p>
</footer>

</body>
</html>'''
