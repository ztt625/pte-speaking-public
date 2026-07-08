#!/usr/bin/env python3
"""
小狐泥PTE口语诊断 — FastAPI 轻量版
替代 Gradio，省内存跑 Render 512MB
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

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

from analyze_v2 import run_analysis
from export_html import build_single_student_html

app = FastAPI(title="小狐泥PTE口语诊断")

PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>小狐泥PTE口语诊断</title>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;background:#fffdf8;color:#1a1a2e;min-height:100vh}
  .container{max-width:600px;margin:0 auto;padding:24px 16px}
  h1{font-size:24px;margin-bottom:4px} h1 span{color:#e8870a}
  .tagline{font-size:14px;color:#888;margin-bottom:20px}
  label{display:block;font-weight:600;font-size:14px;margin:14px 0 6px}
  input[type=file]{width:100%;padding:10px;border:1px solid #e8870a;border-radius:8px;background:#fff;font-size:14px}
  textarea{width:100%;padding:10px;border:1px solid #e8870a;border-radius:8px;font-size:14px;resize:vertical;font-family:inherit}
  button{width:100%;padding:12px;background:#e8870a;color:#fff;border:none;border-radius:8px;font-size:16px;font-weight:600;cursor:pointer;margin-top:16px}
  button:hover{opacity:.85} button:disabled{opacity:.5;cursor:not-allowed}
  #result{margin-top:20px}
  .card{background:#fff;border:1px solid #e8870a;border-radius:10px;padding:18px;margin-bottom:12px}
  .card h3{font-size:15px;margin-bottom:8px} .card p{font-size:13px;color:#666}
  .error{color:#c00;font-size:13px}
  .spinner{display:none;text-align:center;padding:20px;color:#888;font-size:14px}
  .spinner.active{display:block}
  footer{text-align:center;padding:20px 0;font-size:11px;color:#bbb;margin-top:24px}
</style>
</head>
<body>
<div class="container">
  <h1>🦊 <span>小狐泥</span> PTE 口语诊断</h1>
  <p class="tagline">上传录音，AI 分析你的口语问题</p>
  <form id="form" enctype="multipart/form-data">
    <label>🎙️ 上传录音（最长 2 分钟）</label>
    <input type="file" name="audio" accept="audio/*" required>
    <label>📖 原文（可选）</label>
    <textarea name="text" placeholder="粘贴原文，诊断更精准..." rows="2"></textarea>
    <button type="submit">🔍 开始诊断</button>
  </form>
  <div class="spinner" id="spinner">⏳ 分析中，约需 1-2 分钟...</div>
  <div id="result"></div>
</div>
<footer>
  <p>⚡ 当前为免费公测版，基于 AI 模型自动分析，结果仅供参考。</p>
</footer>
<script>
document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  var btn = e.target.querySelector('button');
  var spinner = document.getElementById('spinner');
  var result = document.getElementById('result');
  btn.disabled = true;
  spinner.classList.add('active');
  result.innerHTML = '';
  var fd = new FormData(e.target);
  try {
    var resp = await fetch('/diagnose', { method: 'POST', body: fd });
    if (!resp.ok) {
      var err = await resp.json();
      result.innerHTML = '<div class="card"><h3>❌ 分析失败</h3><p class="error">' + err.detail + '</p></div>';
    } else {
      var blob = await resp.blob();
      var url = URL.createObjectURL(blob);
      var summary = resp.headers.get('X-Summary') || '';
      result.innerHTML = '<div class="card"><h3>✅ 诊断完成</h3><p>' + decodeURIComponent(summary) + '</p><p style="margin-top:10px"><a href="' + url + '" download="诊断报告.html" style="color:#e8870a;font-weight:700">📄 下载诊断报告</a></p></div>';
    }
  } catch (err) {
    result.innerHTML = '<div class="card"><h3>❌ 网络错误</h3><p class="error">请检查网络后重试</p></div>';
  } finally {
    btn.disabled = false;
    spinner.classList.remove('active');
  }
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE_HTML


@app.post("/diagnose")
async def diagnose(audio: UploadFile = File(...), text: str = Form("")):
    suffix = Path(audio.filename).suffix if audio.filename else ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(await audio.read())
        tmp.close()
    except Exception:
        os.unlink(tmp.name)
        return JSONResponse({"detail": "音频保存失败"}, 400)

    text_clean = text.strip() or None

    try:
        result = run_analysis(
            audio_path=tmp.name, ref_path=None,
            text=text_clean, model_name="tiny",
            student_name="同学", question_type="RA",
        )
    except Exception as e:
        os.unlink(tmp.name)
        return JSONResponse({"detail": str(e)}, 500)

    export_data = {
        "student_name": "同学", "audio_path": tmp.name,
        "result": result, "markdown": result["markdown"],
        "question_type": "RA", "text": text_clean or "",
    }
    try:
        html_path = build_single_student_html(export_data)
    except Exception as e:
        os.unlink(tmp.name)
        return JSONResponse({"detail": str(e)}, 500)

    summary = (
        f"语速: {result['rhythm']['wpm']} WPM | "
        f"蹦词指数: {result['rhythm']['staccato_score']}/6 | "
        f"虚实比: {result['rhythm']['content_func_ratio']}:1"
    )

    os.unlink(tmp.name)
    return FileResponse(html_path, media_type="text/html",
                        filename="诊断报告.html",
                        headers={"X-Summary": summary})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
