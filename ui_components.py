#!/usr/bin/env python3
"""PTE 口语诊断 — UI 组件函数
============================
纯函数，无 Gradio 依赖，负责数据表格、节奏详情和交互式波形图生成。
"""

import os
import re
import io
import json
import base64
import numpy as np
import pandas as pd
import librosa


def build_comparison_df(comparison):
    """将对比结果转成 Gradio DataFrame 格式"""
    if not comparison or not comparison.get("word_comparisons"):
        return pd.DataFrame({"提示": ["未提供参考录音，无对比数据"]})

    rows = []
    for c in comparison["word_comparisons"]:
        s_dur = f"{c['student_dur']}s" if c.get("student_dur") is not None else "—"
        n_dur = f"{c['native_dur']}s" if c.get("native_dur") is not None else "—"
        diff = f"{c['diff_ms']}ms" if c.get("diff_ms") is not None else "—"
        ratio = f"{c['ratio']}x" if c.get("ratio") is not None else "—"

        note = ""
        sa = c.get("student_heard_as")
        na = c.get("native_heard_as")
        if sa and sa != c["word"]:
            note += f"生→'{sa}' "
        if na and na != c["word"]:
            note += f"师→'{na}' "

        rows.append({
            "词": c["word"],
            "学生时长": s_dur,
            "参考时长": n_dur,
            "差异": diff,
            "比例": ratio,
            "判断": c["verdict"],
            "备注": note.strip(),
        })

    return pd.DataFrame(rows)


def format_rhythm_detail(rhythm, pitch, comparison):
    """格式化节奏/语调详情为 Markdown"""
    lines = []
    lines.append("### 📊 基础指标")
    lines.append(f"- **语速**: {rhythm['wpm']} WPM ({rhythm['wpm_verdict']})")
    lines.append(f"- **词间平均间隙**: {rhythm['mean_gap_ms']}ms")
    lines.append(f"- **虚实词时长比**: {rhythm['content_func_ratio']}:1 ({rhythm['ratio_verdict']})")
    lines.append(f"- **词长变异系数**: {rhythm['dur_cv']} (自然区间 0.3–0.55)")
    lines.append(f"- **蹦词指数**: {rhythm['staccato_score']}/6 → {rhythm['staccato_verdict']}")
    lines.append("")

    if rhythm["staccato_reasons"]:
        lines.append("**蹦词/等时节奏证据**:")
        for r in rhythm["staccato_reasons"]:
            lines.append(f"- {r}")
        lines.append("")

    if rhythm.get("dragged_func"):
        lines.append("**虚词拖音**:")
        for d in rhythm["dragged_func"]:
            lines.append(f"- '{d['word']}' {d['duration']}s (均值 {d['mean_dur']}s, {d['over_ratio']}倍 → 拖虚词想实词)")
        lines.append("")
    if rhythm.get("dragged_content"):
        lines.append("**实词拖音**:")
        for d in rhythm["dragged_content"]:
            lines.append(f"- '{d['word']}' {d['duration']}s (均值 {d['mean_dur']}s, {d['over_ratio']}倍 → 元音/重音拉得过长)")
        lines.append("")

    lines.append("### 🎤 语调")
    lines.append(f"- **基频均值**: {pitch['mean_f0']}Hz (范围 {pitch['min_f0']}–{pitch['max_f0']}Hz)")
    lines.append(f"- **音高变化**: {pitch['variation']} (CV={pitch.get('variation_cv', '?')}%)")
    lines.append(f"- **句尾趋势**: {pitch['tail_trend']} — {pitch.get('tail_verdict', '?')}")
    lines.append("")

    if comparison and comparison.get("pitch_comparison"):
        pc = comparison["pitch_comparison"]
        lines.append("### ⚖️ 音高对比")
        lines.append(f"- 学生基频: {pc['student_mean_f0']}Hz | 参考: {pc['native_mean_f0']}Hz")
        lines.append(f"- 学生音高范围: {pc['student_range']}Hz | 参考: {pc['native_range']}Hz")

    return "\n".join(lines)


def build_word_diag_df(word_diag):
    """将逐词诊断转为 DataFrame"""
    if not word_diag:
        return pd.DataFrame({"提示": ["暂无数据"]})

    rows = []
    for w in word_diag:
        main_issue = ""
        for issue in w["issues"]:
            if issue != "✓ 正常":
                main_issue = issue
                break
        if not main_issue:
            main_issue = "✓ 正常"

        ref_str = f"{w['ref_duration']}s" if w.get("ref_duration") else "—"
        dur = w.get("duration")
        dur_str = f"{dur}s" if dur is not None else "—"
        prob = w.get("prob")
        prob_str = f"{prob:.0%}" if prob is not None else "—"

        # 对齐状态
        match = w.get("match", "")
        if match == "missing":
            match_str = "🔴 缺失"
        elif match in ("substitution", "fuzzy"):
            match_str = "🔴 替换"
        elif match == "extra":
            match_str = "⚠️ 多读"
        elif match in ("exact", "fuzzy"):
            match_str = "✓"
        else:
            match_str = "—"

        # 替换词显示原文 → 识别
        word_display = w["word"]
        if w.get("whisper_word") and w["whisper_word"] != w["word"]:
            word_display = f"{w['word']} → {w['whisper_word']}"

        rows.append({
            "#": w["index"] + 1,
            "词": word_display,
            "时长": dur_str,
            "置信度": prob_str,
            "参考": ref_str,
            "词性": "虚词" if w["is_function_word"] else "实词",
            "对齐": match_str,
            "诊断": main_issue,
            "严重度": {"ok": "✓", "warning": "⚠️", "error": "🔴"}.get(w["severity"], "?"),
        })

    return pd.DataFrame(rows)


def build_interactive_waveform_html(audio_path, result, wrap_iframe=True):
    """生成交互式波形图 HTML — 音频 + 波形同步，点击跳转，标注叠加。

    wrap_iframe=True : 返回 iframe 包裹的独立页面（用于 gr.HTML 组件）
    wrap_iframe=False: 返回带 data 属性的 div（用于嵌入 markdown，需配合全局 JS）
    出错时自动回退到静态波形图。
    """
    # ── 回退：无音频文件 ──
    if not audio_path or not os.path.exists(audio_path):
        wf = result.get("waveform_b64", "")
        if wf:
            return f'<img src="data:image/png;base64,{wf}" style="max-width:100%;" alt="波形图">'
        return "<p>⚠️ 波形图不可用</p>"

    try:
        # ── 1. 提取波形峰值 ──
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        total_dur = len(y) / sr
        num_peaks = 600
        hop = max(len(y) // num_peaks, 1)
        peaks = []
        for i in range(0, len(y), hop):
            chunk = y[i:i + hop].astype(np.float64)
            peaks.append(float(np.sqrt(np.mean(chunk ** 2))))
        mx = max(peaks) if peaks else 1.0
        if mx > 0:
            peaks = [round(p / mx, 4) for p in peaks]

        # ── 2. 音频 → 低码率 base64（防 data URL 超浏览器限制）──
        # 根据时长动态选采样率，确保 data URL < 2MB
        import wave as _wave
        if total_dur <= 30:
            PLAYER_SR = 8000   # 短音频保持高音质
        else:
            PLAYER_SR = 6000   # 长音频降到 6kHz，2分钟 ≈ 1.4MB base64
        if sr != PLAYER_SR:
            y_p = librosa.resample(y.astype(np.float64), orig_sr=sr, target_sr=PLAYER_SR)
        else:
            y_p = y
        pk = np.max(np.abs(y_p)) or 1.0
        y_pcm = (y_p / pk * 0.9 * 32767).astype(np.int16)
        _buf = io.BytesIO()
        with _wave.open(_buf, 'wb') as _wf:
            _wf.setnchannels(1)
            _wf.setsampwidth(2)
            _wf.setframerate(PLAYER_SR)
            _wf.writeframes(y_pcm.tobytes())
        _buf.seek(0)
        audio_b64 = base64.b64encode(_buf.read()).decode()
        mime = "wav"

        # ── 3. 词级标注（从 whisper 时间戳，索引已由 analyze_v2 对齐）──
        whisper_words = result.get("whisper_result", {}).get("words", [])
        word_diag = result.get("word_diag", [])
        annotations = []
        for w in word_diag:
            idx = w.get("index", 0)
            if idx < len(whisper_words):
                sw = whisper_words[idx]
                s = sw.get("start")
                e = sw.get("end")
                if s is None or e is None or s == e:
                    continue
            else:
                continue
            d = total_dur or 1.0
            annotations.append({
                "w": w.get("word", ""),
                "s": round(s / d, 4),
                "e": round(e / d, 4),
                "sv": w.get("severity", "ok"),
            })

        peaks_json = json.dumps(peaks)
        ann_json = json.dumps(annotations, ensure_ascii=False)
        dur_fmt = f"{int(total_dur // 60)}:{int(total_dur % 60):02d}"

        # ── 波形下方文字：Whisper 识别文本 + 重点发音偏差 ──
        #    SGD 等长音频 Whisper 常不输出标点 → 自动补标点让文本可读
        whisper_text = result.get("whisper_result", {}).get("text", "")
        if whisper_text:
            from analyze_v2 import _restore_punctuation
            whisper_text, _ = _restore_punctuation(whisper_text)
        dev_items = []
        for w in word_diag:
            if w.get("severity") == "error":
                issue = ""
                for iss in w.get("issues", []):
                    if iss != "✓ 正常":
                        issue = iss
                        break
                dict_uri = w.get("dict_audio_uri", "")
                dev_items.append((w.get("word", ""), issue, dict_uri))
        dev_short = dev_items[:8] if len(dev_items) <= 8 else dev_items[:6]

        caption_parts = []
        if whisper_text:
            caption_parts.append(
                f'<div style="margin-top:10px;padding:8px 12px;background:#2d2d2d;color:#e0e0e0;'
                f'border-left:3px solid #e67e22;border-radius:4px;font-size:13px;line-height:1.6;">'
                f'<b>📝 识别文本：</b>{whisper_text}</div>'
            )
        if dev_short:
            dev_parts = []
            for w, iss, dict_uri in dev_short:
                if dict_uri:
                    audio_btn = (
                        '<span onclick="(function(){'
                        'var a=new Audio(\'' + dict_uri + '\');'
                        'a.play();})()" '
                        'style="cursor:pointer;font-size:15px;margin-right:1px;'
                        'filter:brightness(1.2);" '
                        'title="点击听标准发音（TTS）">🔊</span>'
                    )
                else:
                    audio_btn = (
                        '<span style="font-size:13px;margin-right:1px;opacity:0.4;" '
                        'title="标准发音暂不可用">🔇</span>'
                    )
                issue_suffix = f'<span style="color:#c0a0a0;font-size:11px;">: {iss}</span>' if iss else ""
                dev_parts.append(
                    '<span style="display:inline-flex;align-items:center;margin:3px 8px 3px 0;'
                    'padding:4px 10px;background:#2d1a1a;border:1px solid #6b2c2c;'
                    'border-radius:5px;font-size:12px;gap:4px;">'
                    + audio_btn +
                    f'<b style="color:#f07070;">{w}</b>'
                    + issue_suffix +
                    '</span>'
                )
            dev_lines = "".join(dev_parts)
            more = (
                f' <span style="color:#888;font-size:11px;">+{len(dev_items) - len(dev_short)} 词</span>'
                if len(dev_items) > len(dev_short) else ""
            )
            caption_parts.append(
                '<div style="margin-top:8px;padding:12px 14px;line-height:1.8;'
                'background:#1a1a1a;border:1px solid #333;border-radius:8px;">'
                '<b style="color:#e67e22;font-size:13px;">🔴 发音偏差较大：</b>'
                '<div style="display:flex;flex-wrap:wrap;align-items:center;margin-top:4px;">'
                + dev_lines + more +
                '</div></div>'
            )
        caption_html = "\n".join(caption_parts)

        if wrap_iframe:
            # ── iframe 模式：自包含页面 ──
            import random as _random
            import string as _string
            uid = ''.join(_random.choices(_string.ascii_lowercase, k=8))

            page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#1e1e1e;font-family:-apple-system,BlinkMacSystemFont,sans-serif;padding:10px 12px;overflow:hidden;}}
canvas{{display:block;width:100%;height:180px;cursor:pointer;border-radius:4px;}}
.ctls{{display:flex;align-items:center;gap:10px;margin-top:8px;color:#ccc;flex-wrap:wrap;}}
button{{background:#e67e22;border:none;color:#fff;padding:5px 14px;border-radius:4px;cursor:pointer;font-size:13px;font-weight:600;}}
button:hover{{background:#d35400;}}
.tm{{font-size:12px;font-family:monospace;}}
.ht{{font-size:11px;color:#888;margin-left:4px;}}
.lg{{display:inline-flex;align-items:center;gap:3px;font-size:10px;color:#aaa;}}
.ld{{display:inline-block;width:8px;height:8px;border-radius:2px;}}
.tt{{position:absolute;background:rgba(0,0,0,0.85);color:#fff;padding:3px 7px;border-radius:4px;font-size:11px;pointer-events:none;white-space:nowrap;display:none;z-index:10;}}
</style></head>
<body>
<canvas id="c" width="800" height="180"></canvas>
<div class="tt" id="tt"></div>
<div class="ctls">
  <button id="b" onclick="tgl()">▶ 播放</button>
  <span class="tm" id="t">0:00 / {dur_fmt}</span>
  <span class="ht">| 点击波形图跳转 |</span>
  <span class="lg"><span class="ld" style="background:#e74c3c;"></span>严重</span>
  <span class="lg" style="margin-left:6px;"><span class="ld" style="background:#f39c12;"></span>注意</span>
</div>
<audio id="a" src="data:audio/{mime};base64,{audio_b64}" preload="auto" style="display:none;"></audio>
<script>
var P={peaks_json};
var A={ann_json};
var D={total_dur};
var cv=document.getElementById('c');
var ad=document.getElementById('a');
var bt=document.getElementById('b');
var td=document.getElementById('t');
var ctx=cv.getContext('2d');
var W=cv.width,H=cv.height;
var playing=false;

function fmt(t){{var m=Math.floor(t/60),s=Math.floor(t%60);return m+':'+(s<10?'0':'');}}

function draw(php){{
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#252525';ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#3a3a3a';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(0,H/2);ctx.lineTo(W,H/2);ctx.stroke();
  ctx.strokeStyle='#333';ctx.lineWidth=0.5;
  for(var i=0;i<W;i+=80){{ctx.beginPath();ctx.moveTo(i,0);ctx.lineTo(i,H);ctx.stroke();}}
  var bw=W/P.length;
  for(var i=0;i<P.length;i++){{
    var x=i*bw,bh=P[i]*(H/2-8);
    ctx.fillStyle=(php!==undefined&&i/P.length<php)?'#e67e22':'#666';
    ctx.fillRect(x,H/2-bh,Math.max(bw-0.5,0.5),bh*2);
  }}
  for(var j=0;j<A.length;j++){{
    var a=A[j],ax=a.s*W,aw=Math.max((a.e-a.s)*W,2);
    ctx.fillStyle=a.sv==='error'?'rgba(231,76,60,0.25)':a.sv==='warning'?'rgba(243,156,18,0.2)':'transparent';
    if(ctx.fillStyle!=='transparent')ctx.fillRect(ax,0,aw,H);
    ctx.save();
    ctx.translate(ax+Math.max(aw/2,1), H-2);
    ctx.rotate(-Math.PI/2);
    ctx.textBaseline='bottom';
    var c=a.sv==='error'?'#e74c3c':a.sv==='warning'?'#f39c12':'#888';
    ctx.fillStyle=c;ctx.font='8px system-ui';ctx.fillText(a.w,0,0);
    ctx.restore();
  }}
  if(php!==undefined){{
    var px=php*W;
    ctx.strokeStyle='#e67e22';ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(px,0);ctx.lineTo(px,H);ctx.stroke();
    ctx.fillStyle='#e67e22';
    ctx.beginPath();ctx.moveTo(px,0);ctx.lineTo(px-5,8);ctx.lineTo(px+5,8);ctx.fill();
  }}
}}

function upd(){{
  if(ad.duration&&ad.duration>0){{var p=ad.currentTime/ad.duration;draw(p);td.textContent=fmt(ad.currentTime)+' / '+fmt(D);}}
  if(playing)requestAnimationFrame(upd);
}}

function tgl(){{
  if(playing){{ad.pause();playing=false;bt.textContent='▶ 播放';}}
  else{{ad.play();playing=true;bt.textContent='⏸ 暂停';upd();}}
}}

ad.onended=function(){{playing=false;bt.textContent='▶ 播放';draw(1.0);td.textContent=fmt(D)+' / '+fmt(D);}};
ad.onloadedmetadata=function(){{D=ad.duration||D;td.textContent='0:00 / '+fmt(D);}};
cv.onclick=function(e){{var r=cv.getBoundingClientRect();var p=(e.clientX-r.left)/r.width;if(ad.duration)ad.currentTime=p*ad.duration;draw(p);if(!playing)td.textContent=fmt(p*D)+' / '+fmt(D);}};
var tt=document.getElementById('tt');
cv.onmousemove=function(e){{var r=cv.getBoundingClientRect();var sx=W/r.width;var mx=(e.clientX-r.left)*sx,my=e.clientY-r.top;var found=null;for(var j=0;j<A.length;j++){{var a=A[j],ax=a.s*W,aw=Math.max((a.e-a.s)*W,2);if(mx>=ax&&mx<=ax+aw){{found=a;break;}}}}if(found){{var sv={{'error':'🔴严重','warning':'🟡注意','ok':'✓正常'}}[found.sv]||found.sv;tt.style.display='block';tt.style.left=(mx/sx+10)+'px';tt.style.top=(my-26)+'px';tt.textContent=found.w+' — '+sv;}}else{{tt.style.display='none';}}}};
cv.onmouseout=function(){{tt.style.display='none';}};
draw();
setTimeout(function(){{if(ad.duration&&ad.duration>0){{D=ad.duration;td.textContent='0:00 / '+fmt(D);}}}},800);
</script>
</body>
</html>"""

            encoded = base64.b64encode(page.encode('utf-8')).decode()
            return (
                f'<iframe src="data:text/html;base64,{encoded}" '
                f'style="width:100%;height:255px;border:none;border-radius:8px;" '
                f'sandbox="allow-scripts allow-same-origin"></iframe>\n'
                f'{caption_html}'
            )
        else:
            # ── 嵌入模式：div + data 属性（用于 markdown，由全局 JS 驱动）──
            peaks_b64 = base64.b64encode(peaks_json.encode()).decode()
            ann_b64 = base64.b64encode(ann_json.encode()).decode()
            return (
                f'<div class="iwf-root" data-iwf-peaks="{peaks_b64}" data-iwf-ann="{ann_b64}" data-iwf-dur="{total_dur}">'
                f'<div style="background:#1e1e1e;border-radius:8px;padding:10px 12px;'
                f'font-family:-apple-system,BlinkMacSystemFont,sans-serif;">'
                f'<canvas width="800" height="180" style="width:100%;height:180px;cursor:pointer;'
                f'border-radius:4px;display:block;"></canvas>'
                f'<div style="display:flex;align-items:center;gap:10px;margin-top:8px;color:#ccc;flex-wrap:wrap;">'
                f'<button class="iwf-play" style="background:#e67e22;border:none;color:#fff;padding:5px 14px;'
                f'border-radius:4px;cursor:pointer;font-size:13px;font-weight:600;">▶ 播放</button>'
                f'<span class="iwf-time" style="font-size:12px;font-family:monospace;">0:00 / {dur_fmt}</span>'
                f'<span style="font-size:11px;color:#888;">| 点击波形图跳转 |</span>'
                f'<span style="display:inline-block;width:8px;height:8px;background:#e74c3c;border-radius:2px;"></span>'
                f'<span style="font-size:10px;color:#aaa;">严重</span>'
                f'<span style="display:inline-block;width:8px;height:8px;background:#f39c12;border-radius:2px;margin-left:4px;"></span>'
                f'<span style="font-size:10px;color:#aaa;">注意</span>'
                f'</div></div>'
                f'<audio class="iwf-audio" src="data:audio/{mime};base64,{audio_b64}" preload="auto" style="display:none;"></audio>'
                f'</div>\n'
                f'{caption_html}'
            )

    except Exception:
        wf = result.get("waveform_b64", "")
        if wf:
            return (
                f'<p><em>⚠️ 交互式波形加载失败，回退静态图</em></p>\n'
                f'<img src="data:image/png;base64,{wf}" style="max-width:100%;" alt="波形图">'
            )
        return "<p>⚠️ 波形图生成失败</p>"
