#!/usr/bin/env python3
"""
PTE 口语诊断 — 批量分析引擎 v2.0
===============================
多题并行分析 + 跨题型综合报告。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from concurrent.futures import ThreadPoolExecutor, as_completed
from analyze_v2 import run_analysis

QUESTION_TYPES = ["RA", "RS", "DI", "RL", "SGD", "RTS"]


def run_batch(items, model_name="base", student_name=""):
    """
    批量分析多道题。
    items: [{type, audio, text, ref?}]
    返回: {per_question, synthesis}
    """
    results = [None] * len(items)  # 保持顺序

    def analyze_one(idx, item):
        try:
            r = run_analysis(
                audio_path=item["audio"],
                ref_path=item.get("ref"),
                text=item.get("text") or None,
                model_name=model_name,
                student_name=student_name,
                question_type=item.get("type", "RA"),
            )
            return idx, r, None
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            # 截取最后几行（最相关的错误信息）
            tb_lines = tb.strip().split("\n")
            short_tb = "\n".join(tb_lines[-4:]) if len(tb_lines) > 4 else tb
            return idx, None, f"{type(e).__name__}: {e}\n```\n{short_tb}\n```"

    # 并行分析（最多 4 题同时）
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(analyze_one, i, item): i for i, item in enumerate(items)}
        for f in as_completed(futures):
            idx, result, error = f.result()
            if error:
                results[idx] = {"error": error, "type": items[idx].get("type", "RA")}
            else:
                results[idx] = result

    synthesis = generate_synthesis(results, student_name)
    return {"per_question": results, "synthesis": synthesis}


def generate_synthesis(results, student_name=""):
    """聚合所有单题结果，生成综合诊断"""
    if not results:
        return {"error": "No results"}

    valid = [r for r in results if r and "error" not in r]
    if not valid:
        return {"error": "All analyses failed"}

    # ── 错误类型统计 ──
    error_counter = {}  # error_label → count
    question_errors = {}  # question_label → [errors]

    for i, r in enumerate(results):
        if not r or "error" in r:
            continue
        qtype = r.get("question_type", "RA")
        label = f"{qtype}#{i+1}"
        q_errors = []
        flags = r.get("flags", {})

        for dim, items_list in flags.items():
            for item in items_list:
                # 提取错误关键词
                key = _extract_error_key(item)
                if key:
                    error_counter[key] = error_counter.get(key, 0) + 1
                    q_errors.append(key)

        # 逐词诊断统计
        for w in r.get("word_diag", []):
            if w["severity"] != "ok":
                for issue in w["issues"]:
                    if issue != "✓ 正常":
                        key = _extract_error_key(issue)
                        if key and key not in q_errors:
                            q_errors.append(key)
                            error_counter[key] = error_counter.get(key, 0) + 1

        question_errors[label] = q_errors

    # 排序
    ranked = sorted(error_counter.items(), key=lambda x: -x[1])

    # ── 指标总览 ──
    metrics_table = []
    for i, r in enumerate(results):
        if not r or "error" in r:
            continue
        qtype = r.get("question_type", "RA")
        rh = r.get("rhythm", {})
        pi = r.get("pitch", {})
        metrics_table.append({
            "题号": f"{qtype}#{i+1}",
            "语速WPM": rh.get("wpm", "—"),
            "虚实比": f"{rh.get('content_func_ratio', '—')}:1",
            "蹦词分": f"{rh.get('staccato_score', 0)}/6",
            "卡顿": rh.get("hesitation_count", 0),
            "音高": pi.get("variation", "—"),
            "能量": pi.get("energy_stability", "—"),
        })

    # ── 热力图数据 ──
    error_types = ["蹦词/等时节奏", "虚词拖音", "吞音/含糊", "元音/辅音不准",
                   "卡顿", "语调异常", "意群断裂", "拖音", "语速问题"]
    heatmap = {}
    for i, r in enumerate(results):
        if not r or "error" in r:
            continue
        qtype = r.get("question_type", "RA")
        label = f"{qtype}#{i+1}"
        row = {}
        for et in error_types:
            row[et] = _count_error_type(r, et)
        heatmap[label] = row

    # ── 一句话总结 ──
    summary = _generate_summary(ranked, question_errors, len(valid))

    return {
        "student_name": student_name,
        "total_questions": len(results),
        "valid_count": len(valid),
        "failed_count": len(results) - len(valid),
        "top_errors": ranked[:5],
        "question_errors": {k: list(set(v)) for k, v in question_errors.items()},
        "metrics_table": metrics_table,
        "heatmap": heatmap,
        "error_types": error_types,
        "summary": summary,
    }


def _extract_error_key(text):
    """从诊断文本提取简短错误标签"""
    keywords = {
        "蹦词": "蹦词/等时节奏",
        "等时节奏": "蹦词/等时节奏",
        "虚词拖": "虚词拖音",
        "拖音": "拖音",
        "句尾拖音": "拖音",
        "吞音": "吞音/含糊",
        "含糊": "吞音/含糊",
        "元音不饱满": "元音/辅音不准",
        "辅音不准": "元音/辅音不准",
        "发音偏模糊": "元音/辅音不准",
        "卡顿": "卡顿",
        "停顿": "意群断裂",
        "间隙": "意群断裂",
        "意群断裂": "意群断裂",
        "在虚词后断了": "意群断裂",
        "上扬": "语调异常",
        "语调": "语调异常",
        "音高偏高": "语调异常",
        "音高偏低": "语调异常",
        "语速偏慢": "语速问题",
        "语速偏快": "语速问题",
        "实词太短": "语速问题",
    }
    for kw, label in keywords.items():
        if kw in text:
            return label
    # Fallback: use first 10 chars
    if len(text) > 10:
        return text[:10]
    return text


def _count_error_type(result, error_type):
    """统计特定错误类型在单题中的严重度（0-3）"""
    count = 0
    flags = result.get("flags", {})
    for dim, items in flags.items():
        for item in items:
            if _extract_error_key(item) == error_type:
                count += 1 if "🔴" in item else 0.5
    for w in result.get("word_diag", []):
        for issue in w.get("issues", []):
            if _extract_error_key(issue) == error_type:
                count += 0.5 if w["severity"] == "error" else 0.25
    return min(round(count, 1), 3)


def _generate_summary(ranked, question_errors, total):
    """生成一句话诊断总结"""
    if not ranked:
        return "无法生成总结"

    top = [f"{label}({count}次)" for label, count in ranked[:3]]
    parts = [f"共性根因TOP{len(top)}: {', '.join(top)}"]

    # 按题型分组
    by_type = {}
    for q_label, errors in question_errors.items():
        qtype = q_label.split("#")[0]
        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].extend(errors)

    type_summaries = []
    for qtype in ["RA", "RS", "DI", "RL", "SGD", "RTS"]:
        if qtype in by_type and by_type[qtype]:
            top_err = max(set(by_type[qtype]), key=by_type[qtype].count)
            type_summaries.append(f"{qtype}中{top_err}最突出")

    if type_summaries:
        parts.append("；".join(type_summaries[:3]))

    return "。".join(parts) + "。"


def build_synthesis_markdown(synthesis):
    """生成综合报告 Markdown — 老师友好版。"""
    if synthesis.get("error"):
        return f"❌ {synthesis['error']}"

    lines = []
    name = synthesis.get("student_name", "学生")
    lines.append(f"# {name} — PTE 口语综合诊断")
    lines.append("")
    lines.append(f"**分析题数**: {synthesis['total_questions']} | "
                 f"**成功**: {synthesis['valid_count']} | "
                 f"**失败**: {synthesis['failed_count']}")
    lines.append("")

    # ── 一句话总结 ──
    lines.append("## 📝 综合总结")
    lines.append(f"> {synthesis['summary']}")
    lines.append("")

    # ── 高频根因（加中文解释）──
    lines.append("## 🔴 最突出的问题")
    top = synthesis.get("top_errors", [])[:3]
    if top:
        desc_map = {
            "蹦词/等时节奏": "词与词之间脱节，像一个个蹦出来的，缺乏连贯感",
            "虚词拖音": "在 the/a/of 等虚词上停留过久，在用虚词想下一个词",
            "吞音/含糊": "发音不够清晰，部分音节被吞掉或模糊带过",
            "元音/辅音不准": "元音没张开或辅音位置不对，语音模型识别成别的词",
            "卡顿": "出现呃/嗯等填充词，或不该停的地方停了半秒以上",
            "语调异常": "音高起伏不自然，可能太平淡或个别词突高突低",
            "意群断裂": "在不应停顿的地方断了，破坏了语义连贯",
            "拖音": "某些词拉得过长，尤其是句尾",
            "语速问题": "整体语速偏快或偏慢",
        }
        for rank, (label, count) in enumerate(top, 1):
            desc = desc_map.get(label, "")
            lines.append(f"{rank}. **{label}**（{count} 次）— {desc}")
    else:
        lines.append("✓ 未检测到明显共性问题")
    lines.append("")

    # ── 各题指标（精简版）──
    lines.append("## 📋 各题结果")
    lines.append("| 题目 | 语速 | 蹦词 | 虚实比 | 主要问题 |")
    lines.append("|------|------|------|--------|---------|")
    if synthesis.get("metrics_table"):
        for row in synthesis["metrics_table"]:
            qlabel = row.get("题号", "?")
            wpm = row.get("语速WPM", "—")
            ratio = row.get("虚实比", "—")
            staccato = row.get("蹦词分", "—")
            # 提取主要问题
            q_errors = synthesis.get("question_errors", {}).get(qlabel, [])
            unique = list(dict.fromkeys(q_errors))[:2]  # 去重取前2
            issues = "、".join(unique) if unique else "✓"
            lines.append(f"| {qlabel} | {wpm} | {staccato} | {ratio} | {issues} |")
    lines.append("")

    # ── 分题型总结 ──
    by_type = {}
    for q_label, errors in synthesis.get("question_errors", {}).items():
        qtype = q_label.split("#")[0]
        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].extend(errors)

    if by_type:
        lines.append("## 🏷 分题型判断")
        for qtype in ["RA", "RS", "DI", "RL", "SGD", "RTS"]:
            if qtype in by_type:
                errs = by_type[qtype]
                if errs:
                    top_err = max(set(errs), key=errs.count)
                    count = errs.count(top_err)
                    lines.append(f"- **{qtype}**：{top_err} 最突出（{count} 次）")
                else:
                    lines.append(f"- **{qtype}**：✓ 无明显问题")
        lines.append("")

    return "\n".join(lines)
