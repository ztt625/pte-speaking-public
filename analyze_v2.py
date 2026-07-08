#!/usr/bin/env python3
"""
PTE 口语诊断 — AI 辅助分析脚本 v2
=====================================
升级内容：
  - 语音识别 base 模型（精度 ↑）
  - 母语者对比模式（--ref native.wav）
  - 中国学习者专项检测（蹦词/虚词拖音/意群/吞音/语调）
  - 输出预填版 Markdown 诊断模板

用法：
  python analyze_v2.py <学生录音.wav> [--ref 母语者录音.wav] [--model base]
"""
import sys
import json
import argparse
import matplotlib
matplotlib.use('Agg')  # 非交互后端，服务端生成图片
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import warnings
from difflib import SequenceMatcher
import numpy as np
import soundfile as sf
import io
import base64
import librosa
from faster_whisper import WhisperModel
import parselmouth
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════

# 虚词集合（检测拖音目标）
FUNCTION_WORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
    "and", "but", "or", "is", "are", "was", "were", "be", "been",
    "that", "this", "these", "those", "it", "its", "has", "have",
    "had", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "do", "does", "did", "not", "as", "no",
    "from", "by", "about", "than", "then", "so", "if", "when",
    "he", "she", "we", "they", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "some", "any", "each", "every",
}


# ══════════════════════════════════════════════════
# CMU Pronouncing Dictionary — 音素级分析
# ══════════════════════════════════════════════════

_cmudict = None

def _get_cmudict():
    """懒加载 CMU 发音词典（~134k 词条，含音素和重音标记）。"""
    global _cmudict
    if _cmudict is None:
        from nltk.corpus import cmudict
        _cmudict = cmudict.dict()
    return _cmudict


# 中国学习者高频出错的音素（CMU 记法）
# 参考：Chinese Speakers' English Pronunciation Problems (Avery & Ehrlich, Swan & Smith)
CN_DIFFICULT_PHONEMES = {
    # 辅音 — 汉语没有
    "TH":  "θ (舌尖齿间清擦音) — 汉语无此音，常被替换为 s/f/t",
    "DH":  "ð (舌尖齿间浊擦音) — 汉语无此音，常被替换为 z/d/l",
    "SH":  "ʃ (清腭龈擦音) — 可能偏 'xī' 感",
    "ZH":  "ʒ (浊腭龈擦音) — 汉语无此音",
    "CH":  "tʃ (清腭龈塞擦音) — 可能偏 'chī' 或 'qī'",
    "JH":  "dʒ (浊腭龈塞擦音) — 可能偏 'zhī' 或 'jī'",
    "R":   "ɹ (龈近音) — 常被替换为类似 r 的卷舌音或 l",
    "L":   "l (边音) — 词尾 dark l 尤其困难，常被省略或变元音",
    "V":   "v (唇齿浊擦音) — 常被替换为 w",
    "Z":   "z (龈浊擦音) — 常被替换为 dz/ts 或清化为 s",
    # 元音 — 汉语元音系统较小，很多英语元音区分困难
    "AE":  "æ (次低前不圆唇) — 常偏 ɛ/e 或 a",
    "IH":  "ɪ (次高前不圆唇) — 常偏 iː，ship→sheep",
    "UH":  "ʊ (次高后圆唇) — 常偏 uː，full→fool",
    "AH":  "ʌ (次低后不圆唇) — 常偏 ɑː 或 ə",
    "ER":  "ɜːr (中央卷舌元音) — 常偏 ə 或儿化不自然",
    "AW":  "aʊ (双元音) — 滑动不充分，偏 'ao'",
    "AY":  "aɪ (双元音) — 滑动不充分，偏 'ai'",
    "EY":  "eɪ (双元音) — 滑动不充分，偏 'ei'",
    "OW":  "oʊ (双元音) — 滑动不充分，偏 'ou'",
    "OY":  "ɔɪ (双元音) — 滑动不充分",
    "NG":  "ŋ (软腭鼻音) — 常加 g 尾音，sing→sing-guh",
}

# 中国学习者难点音素集合（快速查找）
_CN_DIFFICULT_SET = set(CN_DIFFICULT_PHONEMES.keys())



def _get_stress_pattern(word_clean):
    """
    从 CMU 词典提取词的重音模式。
    返回: {
        'syllables': [{'phonemes': [...], 'stress': 0|1|2, 'vowel': 'AH0'}],
        'primary_syllable': 0-based index,
        'n_syllables': int,
        'phoneme_sequence': [...] (flat),
    }
    单音节词或无词条返回 None。
    """
    d = _get_cmudict()
    if word_clean not in d:
        return None
    phonemes = d[word_clean][0]  # 取第一个发音（最常见）

    # 切分音节：元音（含数字 0/1/2 的）是音节核心
    syllables = []
    current = []
    for ph in phonemes:
        current.append(ph)
        if ph[-1].isdigit():
            syllables.append(list(current))
            current = []
    if current and syllables:
        syllables[-1].extend(current)

    # 单音节词也返回 pattern（供音节标注用），但 analyze_word_stress 会跳过
    primary_idx = None
    parsed = []
    for i, syl in enumerate(syllables):
        vowel = next((p for p in syl if p[-1].isdigit()), syl[-1] if syl else "")
        stress = int(vowel[-1]) if vowel[-1].isdigit() else 0
        if stress == 1:
            primary_idx = i
        parsed.append({"phonemes": syl, "stress": stress, "vowel": vowel})

    if primary_idx is None:
        primary_idx = 0  # 兜底：单音节词重音在第0音节

    return {
        "syllables": parsed,
        "primary_syllable": primary_idx,
        "n_syllables": len(parsed),
        "phoneme_sequence": list(phonemes),
    }


def _get_phonemes(word_clean):
    """获取单词的 CMU 音素序列（扁平列表），取第一个发音。失败返回 None。"""
    d = _get_cmudict()
    if word_clean not in d:
        return None
    return list(d[word_clean][0])


def _phonemes_diff(target_clean, recognized_clean):
    """
    对比两个词的 CMU 音素，找出差异。
    用于：当 Whisper 把目标词 A 识别成 B 时，推断是哪个音发错了。
    返回: {
        'target_phonemes': [...],
        'recognized_phonemes': [...],
        'diff_positions': [(pos, target_ph, recog_ph)],
        'target_difficult': [ph for ph in target if ph in CN_DIFFICULT_SET],
        'is_minimal_pair': True if exactly 1-2 phonemes differ,
    }
    """
    tp = _get_phonemes(target_clean)
    rp = _get_phonemes(recognized_clean)
    if not tp or not rp:
        return None

    # 对齐：用 SequenceMatcher 找匹配块
    from difflib import SequenceMatcher as SM2
    sm = SM2(None, tp, rp)
    diff_positions = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                tp_idx = i1 + k if k < (i2 - i1) else i2 - 1
                rp_idx = j1 + k if k < (j2 - j1) else j2 - 1
                diff_positions.append((tp_idx, tp[tp_idx] if tp_idx < len(tp) else "?", rp[rp_idx] if rp_idx < len(rp) else "?"))
        elif tag == "delete":
            for k, idx in enumerate(range(i1, i2)):
                diff_positions.append((idx, tp[idx], "（省略）"))
        elif tag == "insert":
            for k, idx in enumerate(range(j1, j2)):
                diff_positions.append((-1, "（插入）", rp[idx]))

    target_difficult = [ph for ph in tp if ph.rstrip("012") in _CN_DIFFICULT_SET]

    return {
        "target_phonemes": tp,
        "recognized_phonemes": rp,
        "diff_positions": diff_positions,
        "target_difficult": list(set(target_difficult)),
        "is_minimal_pair": len(diff_positions) <= 2,
    }


# ══════════════════════════════════════════════════
# TTS 工具（Microsoft Edge TTS，免费且国内可访问）
# ══════════════════════════════════════════════════

def _edge_tts(text, voice="en-US-AriaNeural"):
    """使用 Microsoft Edge TTS 生成 MP3 字节。
    相比 Google translate_tts（国内被墙），Edge TTS 国内稳定可用。
    返回 bytes，失败返回 None。"""
    import asyncio as _aio
    async def _gen():
        import edge_tts as _et
        chunks = []
        communicate = _et.Communicate(text, voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)
    try:
        return _aio.run(_gen())
    except Exception:
        return None


def _restore_punctuation(text):
    """为无标点文本自动添加句号和逗号。

    适用场景：Whisper 识别结果（尤其是 SGD 等长内容）常缺少标点符号，
    导致 TTS 朗读无停顿、拆句失败。此函数基于语法边界检测，在自然断句处
    插入 . 和 , 使文本更可读、TTS 更自然。

    仅在文本标点密度过低时生效（< 1 个标点 / 80 字符），已有足够标点的
    文本原样返回。

    返回: (带标点的文本, 是否做了修改)
    """
    import re as _re
    if not text or not text.strip():
        return text, False

    # 统计现有标点密度
    punct_chars = len(_re.findall(r'[.!?;:,]', text))
    char_count = len(text)
    # 每 80 字符至少 1 个标点 → 认为标点充足
    if char_count < 80 or punct_chars >= char_count / 80:
        return text, False

    words = text.split()
    n = len(words)
    if n < 8:
        return text, False

    clean = [w.strip(".,;:!?\"'()[]{}").lower() for w in words]

    # ── 语法边界标记（复用 _suggest_sense_groups 的核心逻辑）──
    coordinators = {'and', 'but', 'or', 'nor', 'yet', 'so'}
    subordinators = {
        'because', 'although', 'though', 'while', 'whereas',
        'since', 'unless', 'until', 'after', 'before', 'if', 'when',
        'wherever', 'whenever', 'whatever', 'whether',
    }
    relatives = {'which', 'that', 'who', 'whom', 'whose', 'where'}
    # "and" 后面跟这些词 → 大概率是新句子开头（而非列举项）
    _new_clause_indicators = {
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        'it', 'they', 'he', 'she', 'we', 'there',
        'my', 'your', 'his', 'her', 'our', 'their',
        'its', 'some', 'several', 'each', 'every', 'no',
        'however', 'therefore', 'nevertheless', 'consequently',
        'finally', 'then', 'now', 'still', 'also',
        # 时间过渡词（SGD 口语高频）
        'after', 'before', 'next', 'later', 'once',
        # 补充从句标记
        'what', 'how', 'why', 'where', 'when',
        'another', 'other', 'many', 'most', 'all',
    }

    # ── SGD 口语话语标记（多词）──
    # 在 SGD 自由回答中，学生常用这些短语开启新论点/新句子。
    # 检测到后在前一个词后加句号。
    _discourse_markers = [
        # 开启/列举
        ["to", "start", "with"],
        ["to", "begin", "with"],
        ["first", "of", "all"],
        ["firstly"], ["secondly"], ["thirdly"], ["lastly"],
        # 补充/递进
        ["in", "addition"],
        ["on", "top", "of", "that"],
        ["another", "point", "is"],
        ["another", "reason", "is"],
        ["another", "thing", "is"],
        ["the", "next", "point", "is"],
        # 转折
        ["on", "the", "other", "hand"],
        ["having", "said", "that"],
        # 总结
        ["in", "conclusion"],
        ["to", "sum", "up"],
        ["to", "conclude"],
        ["all", "in", "all"],
        # 过渡
        ["after", "that"],
        ["moving", "on"],
        ["as", "for"],
        ["speaking", "of"],
        ["with", "regard", "to"],
        ["in", "terms", "of"],
        ["when", "it", "comes", "to"],
    ]
    # 最短的话语标记（1-2词），需要严格匹配
    _discourse_markers_single = {
        'firstly', 'secondly', 'thirdly', 'lastly',
        'furthermore', 'moreover', 'additionally',
        'next', 'then', 'finally',
    }

    # period_at[i] = True → 在 word[i] 后面加句号
    period_at = set()
    # comma_at[i] = True → 在 word[i] 后面加逗号
    comma_at = set()

    for i in range(n):
        cw = clean[i]
        raw_w = words[i]

        # 1) 已有标点的不再重复添加
        if any(p in raw_w for p in ['.', ',', ';', ':', '!', '?']):
            continue

        if i < 2 or i >= n - 3:
            continue

        left_words = i + 1
        right_words = n - i - 1

        # 2) 并列连词 → 两边都长 + 后面是新句子开头 → 在连词**前**加句号
        #    区分 "X and Y"（列举，不切）和 "X. And Y..."（独立从句，切）
        #    自适应阈值：短文≥4词，长文按比例放宽
        _coord_min = max(4, n // 10)
        if cw in coordinators and left_words >= _coord_min and right_words >= _coord_min:
            if cw in {'and', 'but', 'or'}:
                next_w = clean[i + 1] if i + 1 < n else ""
                if next_w not in _new_clause_indicators:
                    continue  # 大概率是列举项 (e.g. "Europe and America")
            period_at.add(i - 1)  # 标点在连词之前："...century. And it..."
            continue

        # 3) 从属连词 / 关系代词 → 在之前加逗号
        if cw in subordinators:
            comma_at.add(i - 1)  # "...was harsh, because workers..."
            continue

        if cw in relatives:
            # "that" 作指示代词不切
            if cw == 'that':
                next_w = clean[i + 1] if i + 1 < n else ""
                if next_w in {'the', 'a', 'an', 'this', 'that', 'these',
                              'those', 'my', 'your', 'his', 'her', 'its',
                              'some', 'many', 'much'}:
                    continue
            comma_at.add(i - 1)  # "...transportation, which had..."
            continue

    # ── 4) 话语标记检测（SGD 口语高频）──
    #    学生在自由回答中常用 "to start with", "first of all" 等开启新句子，
    #    在这些话语标记前加句号，让断句更自然。
    for i in range(n):
        if i < 2 or i >= n - 2:
            continue
        # 已被标点覆盖的词跳过
        if i in period_at or i in comma_at:
            continue
        # 多词话语标记匹配（2-4词）
        for dm in _discourse_markers:
            dm_len = len(dm)
            if i + dm_len > n:
                continue
            if clean[i:i+dm_len] == dm:
                # 话语标记前加句号（如果前面有足够内容）
                if i >= 3:
                    period_at.add(i - 1)
                break
        # 单词话语标记（紧随句首或 and/but 之后）
        if clean[i] in _discourse_markers_single:
            if i >= 3:
                period_at.add(i - 1)

    # ── 句号不能太密集（至少间隔 6 个词，长文放宽到 5）──
    _period_spacing = 6 if n <= 40 else 5
    period_list = sorted(period_at)
    filtered = []
    for p in period_list:
        if not filtered or p - filtered[-1] >= _period_spacing:
            filtered.append(p)
    period_at = set(filtered)

    # ── 清理：句号附近的从句逗号（句号后 3 词内不加逗号）──
    #    避免 ". And, after, that..." 这类碎片化
    _period_set = period_at  # 最终的句号位置
    _comma_to_remove = set()
    for cp in comma_at:
        for pp in _period_set:
            if 0 <= cp - pp <= 3:
                _comma_to_remove.add(cp)
                break
    comma_at -= _comma_to_remove

    # ── 逗号不能和句号重叠 ──
    comma_at -= period_at

    if not period_at and not comma_at:
        return text, False

    # ── 重建文本 ──
    result_parts = []
    for i, w in enumerate(words):
        result_parts.append(w)
        if i in period_at:
            result_parts.append('.')
        elif i in comma_at:
            result_parts.append(',')

    result = ' '.join(result_parts)
    # 去掉标点前的空格
    result = _re.sub(r'\s+([.,;:!?])', r'\1', result)
    # 句号后首字母大写
    result = _re.sub(r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper(), result)
    # 确保首字母大写
    if result and result[0].isalpha() and result[0].islower():
        result = result[0].upper() + result[1:]

    return result, True


def _split_text_for_tts(text, max_chars=200):
    """将长文本拆分为适合 TTS 的短句。

    处理有/无标点符号的文本。对于 Whisper 识别结果中常见的
    无标点长文本（如 SGD），回退到语法分句，再回退到按词数均分。

    返回: 句子字符串列表
    """
    import re as _re
    if not text or not text.strip():
        return []

    text = text.strip()

    # ── Level 1: 按句末标点拆分 ──
    parts = _re.split(r'(?<=[.!?])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]

    # ── Level 2: 超长句子按 [;:] 拆分 ──
    expanded = []
    for p in parts:
        if len(p) > max_chars:
            sub = _re.split(r'(?<=[;:])\s+', p)
            expanded.extend([s.strip() for s in sub if s.strip()])
        else:
            expanded.append(p)
    parts = expanded

    # ── Level 3: 仍超长则按逗号拆分 ──
    expanded = []
    for p in parts:
        if len(p) > max_chars:
            sub = _re.split(r'(?<=,)\s+', p)
            expanded.extend([s.strip() for s in sub if s.strip()])
        else:
            expanded.append(p)
    parts = expanded

    # ── Level 4: 仍超长 → 语法边界拆分（修复无标点长文本问题）──
    final = []
    for p in parts:
        if len(p) <= max_chars:
            final.append(p)
        else:
            final.extend(_split_long_clause(p, max_chars))

    # ── 后处理：过短的句子合并到前一句 ──
    merged = []
    for s in final:
        if merged and len(s) < 25:
            merged[-1] = merged[-1] + ' ' + s
        else:
            merged.append(s)

    return merged


def _split_long_clause(text, max_chars=200):
    """对无标点的超长从句，在自然语法边界处拆分。

    优先级：
    1. 并列连词（and, but, or）连接独立从句时
    2. 从属连词（because, although, while...）
    3. 关系代词（which, that, who...）
    4. 兜底：在单词边界按长度均分

    拆分点在连接词**之前**，连接词归入新句开头（更自然）：
      "...the 18th century" / "and it quickly spread..."
    """
    import re as _re
    words = text.split()
    n = len(words)
    if n <= 10:
        return [text]

    clean = [w.strip(".,;:!?\"'()[]{}").lower() for w in words]

    # ── 分句候选标记 ──
    coordinators = {'and', 'but', 'or', 'nor', 'yet', 'so'}
    subordinators = {
        'because', 'although', 'though', 'while', 'whereas',
        'since', 'unless', 'until', 'after', 'before', 'if', 'when',
        'wherever', 'whenever', 'whatever', 'whether',
    }
    relatives = {'which', 'that', 'who', 'whom', 'whose', 'where'}

    # 找候选切分点（切在连接词 i 之前 → 连接词归入右侧新句）
    # split_at = i-1 表示 words[0:i] / words[i:]
    candidates = []
    for i in range(2, n - 3):
        cw = clean[i]
        left_len = len(' '.join(words[:i]))       # 不含连接词
        right_len = len(' '.join(words[i:]))       # 含连接词
        if left_len < 35 or right_len < 35:
            continue

        score = 0
        if cw in coordinators and i >= 4 and (n - i) >= 4:
            score = 10
        elif cw in subordinators:
            score = 8
        elif cw in relatives:
            # "that" 作指示代词（后接名词）不切
            if cw == 'that':
                next_w = clean[i + 1] if i + 1 < n else ""
                if next_w in {'the', 'a', 'an', 'this', 'that', 'these',
                              'those', 'my', 'your', 'his', 'her', 'its',
                              'some', 'many', 'much'}:
                    continue
            score = 6

        if score > 0:
            # split_at = i-1（words[0:i] / words[i:]）
            candidates.append((i - 1, score, i))  # (split_pos, score, connector_pos)

    if not candidates:
        return _split_by_length(text, max_chars)

    # ── 贪心：选最高分切分点，保证段间至少 5 词 ──
    candidates.sort(key=lambda x: -x[1])
    split_points = []
    for split_pos, score, conn_pos in candidates:
        too_close = any(abs(split_pos - sp) < 5 for sp in split_points)
        if too_close:
            continue
        split_points.append(split_pos)
        split_points.sort()

        # 验证所有段都 ≥5 词
        prev = -1
        ok = True
        for sp in split_points:
            if sp - prev < 5:
                ok = False
                break
            prev = sp
        if n - 1 - prev < 5:
            ok = False
        if not ok:
            split_points.remove(split_pos)

    if not split_points:
        return _split_by_length(text, max_chars)

    # ── 构建分段（在 split_pos 后断开，即 words[0:sp+1] / words[sp+1:]）──
    segments = []
    prev = 0
    for sp in split_points:
        segments.append(' '.join(words[prev:sp + 1]))
        prev = sp + 1
    segments.append(' '.join(words[prev:]))

    # 递归处理仍超长的段
    result = []
    for seg in segments:
        if len(seg) > max_chars:
            result.extend(_split_by_length(seg, max_chars))
        else:
            result.append(seg)
    return result


def _split_by_length(text, max_chars=200):
    """在单词边界按 ~max_chars 均分文本。最后的兜底策略。"""
    words = text.split()
    if not words:
        return [text]
    segments = []
    current = []
    current_len = 0
    for w in words:
        added = len(w) + (1 if current else 0)
        if current and current_len + added > max_chars:
            segments.append(' '.join(current))
            current = [w]
            current_len = len(w)
        else:
            current.append(w)
            current_len += added
    if current:
        segments.append(' '.join(current))
    return segments if segments else [text]


def _sync_punctuation_to_words(words, punctuated_text):
    """将标点恢复后的文本中的标点同步回 words 数组。

    words 来自 Whisper 输出，通常不带标点。punctuated_text 是 _restore_punctuation
    处理后的版本，包含插入的 . 和 , 等。此函数将 punctuated_text 中的标点附加到
    对应位置的 word 上，使报告中的逐词展示也带标点。

    策略：punctuated_text 拆词后与 words 逐词对齐（clean 后比对），
    将标点附加到前一个实词上。
    """
    if not words or not punctuated_text:
        return
    p_words = punctuated_text.split()
    n = len(words)
    # 建立 words clean 序列
    w_clean = [w.get("clean") or w["word"].lower().strip(".,;:!?\"'()[]{}") for w in words]
    # 去除 extra 词的干扰，只对齐非 extra 词
    non_extra_indices = [i for i, w in enumerate(words) if w.get("match") != "extra"]

    pi = 0  # punctuated words index
    for wi in non_extra_indices:
        if pi >= len(p_words):
            break
        pw = p_words[pi]
        pw_clean = pw.lower().strip(".,;:!?\"'()[]{}")
        wc = w_clean[wi]
        if pw_clean == wc:
            # 匹配 → 用带标点的版本替换
            if pw != words[wi]["word"]:
                words[wi]["word"] = pw
            pi += 1
        else:
            # 不匹配：可能是标点恢复引入的额外词（如 "And" → ". And"）
            # 跳过 punctuated_text 中的纯标点 token（实际上不会出现）
            # 尝试向前看：p_words[pi] 可能是标点附着在前一词上
            # 例如 pw="experience." 对应 wc="experience" → 上面已处理
            # 如果 pw_clean 在 w_clean 后面某处，说明中间有缺失
            # 跳到下一个匹配位置
            found = False
            for lookahead in range(1, min(4, len(p_words) - pi)):
                if p_words[pi + lookahead].lower().strip(".,;:!?\"'()[]{}") == wc:
                    pi += lookahead
                    pw2 = p_words[pi]
                    if pw2 != words[wi]["word"]:
                        words[wi]["word"] = pw2
                    pi += 1
                    found = True
                    break
            if not found:
                pi += 1  # 跳过，尝试继续


# ══════════════════════════════════════════════════
# PTE 模板词库（DI/RL 模板 vs 内容比例分析）
# ══════════════════════════════════════════════════

TEMPLATE_WORDS = {
    # ── 结论/总结 ──
    "conclusion", "conclude", "overall", "summary", "summarize", "finally",
    # ── 提及/讨论 ──
    "included", "discussed", "mentioned", "mention", "talked", "talk", "saying", "said",
    # ── 展示/说明 ──
    "according", "shows", "show", "shown", "illustrates", "demonstrates", "indicates",
    "represents", "depicts", "displays", "describes",
    # ── 对比/比较 ──
    "compares", "contrast", "compared", "comparison", "whereas", "meanwhile",
    "similarly", "furthermore", "additionally", "moreover", "besides",
    # ── 图表相关 ──
    "graph", "chart", "diagram", "picture", "image", "figure", "table", "data",
    # ── 最高/最低 ──
    "highest", "lowest", "largest", "smallest", "maximum", "minimum",
    # ── 讲座/演讲 ──
    "speaker", "lecture", "lecturer", "professor", "topic", "discussion",
    "mainly", "focuses", "focusing", "focus", "primarily",
    # ── 过渡/填充 ──
    "basically", "actually", "generally", "firstly", "secondly", "lastly",
    "next", "first", "second", "third", "then",
    # ── 常见模板动词/表达 ──
    "see", "seeing", "believe", "think", "seems", "appears", "clear",
    "obvious", "evident", "noticeable",
}

# 多词模板短语（连续 2-4 词匹配，更准）
TEMPLATE_PHRASES = [
    "according to", "the graph shows", "we can see", "it is clear that",
    "after that", "and then", "in addition", "on the other hand",
    "in conclusion", "to sum up", "more than", "less than",
    "higher than", "lower than", "the number of", "the percentage of",
    "the speaker mentioned", "the lecture discussed", "the main point is",
    "looking at", "moving on", "as we can see", "in contrast",
    "compared to", "the highest", "the lowest", "i think",
    "there is", "there are", "this is about", "can be seen",
    "as you can see", "we have", "you can see", "talks about",
    "is about", "the picture shows", "the next one", "going to talk",
    "what we have", "as shown in", "it shows", "this graph",
    "according to the", "from the graph", "based on the",
    "to begin with", "last but not least", "all in all",
]


def analyze_template_content_ratio(whisper_words, question_type="RA"):
    """分析模板词 vs 内容词的比例（DI/RL/SGD 专项）。

    逐词标记 → 统计占比 → 检测连续模板片段 → 给出判定。

    返回 dict 或 None（非 DI/RL 题型）。
    """
    if question_type not in ("DI", "RL", "SGD", "RTS"):
        return None

    raw_words = [w.get("word", "").strip() for w in whisper_words]
    n = len(raw_words)
    if n == 0:
        return None

    # ── 逐词标记模板/内容 ──
    is_template = [False] * n

    # 1) 单词匹配
    for i, w in enumerate(raw_words):
        cw = clean_word(w)
        if cw and cw in TEMPLATE_WORDS:
            is_template[i] = True

    # 2) 短语匹配（sliding window）
    clean_seq = [clean_word(w) for w in raw_words]
    for phrase in TEMPLATE_PHRASES:
        pw = [clean_word(p) for p in phrase.split()]
        pn = len(pw)
        for i in range(n - pn + 1):
            if clean_seq[i:i+pn] == pw:
                for j in range(pn):
                    is_template[i+j] = True

    # ── 统计 ──
    template_count = sum(is_template)
    content_count = n - template_count
    ratio = template_count / n if n > 0 else 0

    total_dur = sum(w.get("duration", 0) or 0 for w in whisper_words)
    template_dur = sum(
        whisper_words[i].get("duration", 0) or 0
        for i in range(n) if is_template[i]
    )
    dur_ratio = template_dur / total_dur if total_dur > 0 else 0

    # ── 连续模板片段（连续 ≥2 个模板词）──
    segments = []
    seg_start = None
    for i in range(n):
        if is_template[i] and seg_start is None:
            seg_start = i
        elif not is_template[i] and seg_start is not None:
            if i - seg_start >= 2:
                phrase = " ".join(raw_words[seg_start:i])
                segments.append((seg_start, phrase))
            seg_start = None
    if seg_start is not None and n - seg_start >= 2:
        phrase = " ".join(raw_words[seg_start:n])
        segments.append((seg_start, phrase))

    # ── 判定 ──
    if ratio < 0.25:
        verdict, icon = "内容为主", "✅"
    elif ratio < 0.40:
        verdict, icon = "均衡", "✅"
    elif ratio < 0.55:
        verdict, icon = "偏模板", "⚠️"
    else:
        verdict, icon = "模板过多", "🔴"

    return {
        "template_word_count": template_count,
        "content_word_count": content_count,
        "total_words": n,
        "template_ratio": round(ratio, 3),
        "template_duration_ratio": round(dur_ratio, 3),
        "verdict": verdict,
        "icon": icon,
        "segments": segments,
    }


def load_audio(path, target_sr=16000):
    """加载音频，统一到 target_sr。
    soundfile 只支持 WAV/FLAC/OGG；MP3/M4A 等格式走 librosa（底层 audioread→ffmpeg）。"""
    path_str = str(path)
    # 先试 soundfile（快，支持 WAV/FLAC/OGG）
    try:
        audio, sr = sf.read(path_str)
    except Exception:
        # 回退 librosa（支持 MP3/M4A/AAC 等，底层 ffmpeg）
        import librosa
        audio, sr = librosa.load(path_str, sr=None, mono=True)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32), target_sr


# ══════════════════════════════════════════════════
# 1. Whisper 转写 + 词级对齐
# ══════════════════════════════════════════════════

import threading
_whisper_models = {}
_whisper_models_lock = threading.Lock()
_whisper_inference_lock = threading.Lock()  # Whisper 模型不支持并行推理，必须串行

def _get_whisper_model(model_name="base"):
    """线程安全的 Whisper 模型缓存（faster-whisper CTranslate2，CPU int8 量化）。"""
    with _whisper_models_lock:
        if model_name not in _whisper_models:
            # int8 量化：CPU 推理速度最快，精度损失可忽略
            _whisper_models[model_name] = WhisperModel(
                model_name, device="cpu", compute_type="int8"
            )
        return _whisper_models[model_name]

def run_whisper(audio, sr, model_name="base"):
    """Whisper 转写，返回词级时间戳（faster-whisper 引擎）"""
    duration = len(audio) / sr if sr > 0 else 0
    if duration < 0.5:
        return {"text": "[录音太短，无法识别]", "words": [], "_error": f"录音仅 {duration:.1f} 秒，语音识别模型需要至少 0.5 秒。请检查录音文件。"}
    MAX_DURATION = 115  # 上限控制在 2 分钟内，留余量避免边界失败
    _truncated = False
    if duration > MAX_DURATION:
        audio = audio[:int(MAX_DURATION * sr)]
        duration = MAX_DURATION
        _truncated = True
    try:
        model = _get_whisper_model(model_name)
        with _whisper_inference_lock:
            segments, _info = model.transcribe(
                audio, word_timestamps=True, language="en",
                beam_size=5, vad_filter=True,
            )
        words = []
        full_text = []
        for seg in segments:
            full_text.append(seg.text.strip())
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word.strip(),
                        "start": w.start,
                        "end": w.end,
                        "duration": w.end - w.start,
                        "prob": w.probability,
                    })
        text = " ".join(full_text)
        if _truncated:
            text = f"⚠️ 录音超过 2 分钟，已自动截取前 {MAX_DURATION} 秒进行分析。\n\n{text}"
        return {"text": text, "words": words}
    except Exception as e:
        return {"text": "[语音识别失败]", "words": [], "_error": f"语音识别转写出错 ({type(e).__name__}): {str(e)[:200]}"}


# ══════════════════════════════════════════════════
# 2. 基频分析（Praat）
# ══════════════════════════════════════════════════

def analyze_pitch(audio, sr, time_step=0.01):
    """提取基频轨迹 + 能量稳定性（PTE 偏好稳定输出）"""
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    pitch_obj = snd.to_pitch(time_step=time_step, pitch_floor=75, pitch_ceiling=600)
    f0 = pitch_obj.selected_array["frequency"]
    times = pitch_obj.xs()

    voiced = f0[f0 > 0]
    if len(voiced) < 5:
        return {"error": "Not enough voiced frames", "mean_f0": 0, "f0_range": 0, "variation": "无数据"}

    cv = np.std(voiced) / np.mean(voiced) * 100

    # 句尾趋势
    n = len(voiced)
    quarter = max(1, n // 4)
    head_mean = np.mean(voiced[:quarter])
    tail_mean = np.mean(voiced[-quarter:])
    slope = (tail_mean - head_mean) / head_mean * 100

    # 能量稳定性（RMS 变异系数）
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    rms_cv = float(np.std(rms) / np.mean(rms) * 100) if np.mean(rms) > 0 else 0
    # PTE 偏好：能量均匀稳定，不忽大忽小
    energy_stability = "稳定" if rms_cv < 50 else "偏波动" if rms_cv < 80 else "不稳定（忽大忽小）"

    # PTE 语调判断（注意：稳定/偏平 = PTE 加分项，不是问题）
    # PTE 不需要夸张抑扬顿挫，需要的是 consistent, controlled delivery
    if cv > 35:
        pitch_verdict = "变化偏大（对PTE来说可能过于戏剧化）"
    elif cv > 18:
        pitch_verdict = "自然（符合PTE要求）"
    else:
        pitch_verdict = "稳定/偏平（符合PTE偏好）"

    return {
        "mean_f0": round(float(np.mean(voiced)), 1),
        "min_f0": round(float(np.min(voiced)), 1),
        "max_f0": round(float(np.max(voiced)), 1),
        "f0_range": round(float(np.max(voiced) - np.min(voiced)), 1),
        "f0_std": round(float(np.std(voiced)), 1),
        "variation_cv": round(float(cv), 1),
        "variation": pitch_verdict,
        "tail_slope_pct": round(float(slope), 1),
        "tail_trend": "上扬↑" if slope > 8 else "降调↓" if slope < -8 else "平稳→",
        "tail_verdict": "⚠️ 陈述句不应上扬" if slope > 8 else "✓ 句尾降调" if slope < -8 else "句尾平稳",
        "energy_cv": round(rms_cv, 1),
        "energy_stability": energy_stability,
        "_f0_array": f0.tolist(),  # 原始基频数组（给逐词分析用）
        "_f0_times": times.tolist(),
    }


def _split_into_sentences(f0, times, audio, sr):
    """基于能量谷底简单切分句子（粗略）"""
    # 简化版：用 RMS 能量找到长停顿
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    # 找能量极低的段落（可能是句间停顿）
    silences = rms < np.median(rms) * 0.3
    # 简化：假设只有一个句子（RA 通常很短）
    voiced_seg = f0[f0 > 0]
    if len(voiced_seg) < 5:
        return []
    return [{
        "mean_f0": round(float(np.mean(voiced_seg)), 1),
        "f0_range": round(float(np.max(voiced_seg) - np.min(voiced_seg)), 1),
    }]


# ══════════════════════════════════════════════════
# 3. 节奏与流利度分析
# ══════════════════════════════════════════════════

def _suggest_sense_groups(original_text):
    """
    Grammar-based sense group segmentation.
    尊重语法完整和语义完整——不在介词短语内部、固定搭配、自然词组中间切分。
    只在大边界切：标点、从句引导词、并列连词连接独立从句、超长意群（>8词）。
    """
    if not original_text:
        return []

    words = original_text.split()
    clean = [w.strip(".,;:!?\"'()[]{}—").lower() for w in words]
    n = len(words)
    boundaries = set()

    # ── 从句引导词 / 关系代词（开启新意群的可靠标志）──
    clause_starters = {
        'because', 'although', 'though', 'if', 'when', 'while',
        'whereas', 'since', 'unless', 'until', 'after', 'before',
        'wherever', 'whenever', 'whatever', 'whether',
        'which', 'that', 'who', 'whom', 'whose', 'where',
        'so',  # so that / so ... that
    }
    # 可作介词或连词的词：后接代词/名词 → 介词，不切；后接从句 → 连词，切
    _preposition_or_conjunction = {'after', 'before', 'since', 'until'}
    # ── 并列连词（连接独立从句时才切）──
    coordinators = {'and', 'but', 'or', 'nor', 'yet'}

    for i in range(n):
        raw_w = words[i]
        cw = clean[i]

        # 1) 标点 → 总是意群边界
        if any(p in raw_w for p in [',', '.', ';', ':', '—', '!', '?']):
            boundaries.add(i)
            continue

        if i > 0 and i < n - 1:
            # 2) 从句引导词 / 关系代词 → 前面断开
            #    排除 "that" 作指示代词的情况（后接名词而非动词）
            if cw in clause_starters:
                # 可作介词的词：后接代词/名词 → 介词短语，不切
                if cw in _preposition_or_conjunction:
                    next_w = clean[i + 1] if i + 1 < n else ""
                    # 后面是代词/限定词 → 介词短语 (after that, before the, since his...)
                    if next_w in {'that', 'this', 'these', 'those', 'the', 'a', 'an',
                                  'my', 'your', 'his', 'her', 'its', 'our', 'their',
                                  'some', 'many', 'each', 'every', 'no', 'any',
                                  'it', 'them', 'him', 'her', 'us', 'me', 'you'}:
                        continue
                if cw == 'that':
                    # "that" 后面是名词/形容词 → 可能是指示代词，不切
                    next_w = clean[i + 1] if i + 1 < n else ""
                    if next_w in {'the', 'a', 'an', 'this', 'that', 'these',
                                  'those', 'my', 'your', 'his', 'her', 'its',
                                  'our', 'their', 'some', 'many', 'much'}:
                        continue  # "that book", "that many people" → 不切
                    # "after that", "before that" → that 不是关系代词，是固定搭配
                    prev_w = clean[i - 1] if i > 0 else ""
                    if prev_w in {'after', 'before', 'since', 'until',
                                  'like', 'and', 'but', 'or', 'with', 'for'}:
                        continue
                if cw == 'so':
                    # "so" 后面是形容词/副词 → "so much", "so quickly" → 不切
                    next_w = clean[i + 1] if i + 1 < n else ""
                    if next_w in {'much', 'many', 'little', 'few', 'far',
                                  'long', 'often', 'quickly', 'slowly'}:
                        continue
                boundaries.add(i - 1)
                continue

            # 3) 并列连词 — 自适应阈值（短文≥4词，长文按比例放宽）
            #    （避免把 "X and Y" 这种名词并列误切）
            if cw in coordinators:
                coord_min = max(4, n // 10)  # 短文≥4，100词文本≥10
                if i >= coord_min and (n - i - 1) >= coord_min:
                    boundaries.add(i - 1)
                continue

    # 4) 长意群补救：自适应阈值 + 分级兜底切分
    #    短文（≤30词）：>12 词才干预；长文：按比例收紧到 n/5
    #    _tight_after 只保留真正不可分的紧密介词：
    #    "of" — most of, one of, results of... 永远不拆
    #    "to" — want to, go to, to do... 紧密补语/不定式
    #    其他介词 (at/in/on/for/with/by/from) 可切——名词+介词短语是天然意群边界
    _tight_after = {'of', 'to'}
    _pronoun_objects = {'me', 'you', 'him', 'her', 'it', 'us', 'them'}
    # 量词/限定词 + 名词 → 不可切 (All students, many people, few things)
    _pre_nominal = {'all', 'many', 'few', 'several', 'both', 'most', 'much',
                    'more', 'some', 'any', 'each', 'every', 'no', 'another'}
    # 二级可接受边界：介词前（名词+介词短语 = 天然意群边界）
    _tier2_prepositions = {'at', 'in', 'on', 'for', 'with', 'by', 'from',
                           'about', 'into', 'onto', 'within', 'without',
                           'during', 'before', 'after', 'between', 'through'}

    # 自适应长意群阈值：
    #   内部 gap：短文≤30词→12，长文→max(10, n/5)
    #   尾段（最后一段）：更宽松——尾段常是总结句，没有明显语法切分点
    _long_gap_threshold = 12 if n <= 30 else max(10, n // 5)
    _tail_threshold = max(15, n // 3)  # 尾段允许更长的连续句

    def _good_boundary(pos, lo, hi, tier=1):
        """检查 pos 是否是好的意群切分点。
        tier=1: 严格（实词后，不破坏紧密结构）
        tier=2: 宽松（接受介词前等次优边界）"""
        if not (lo < pos < hi):
            return False
        if clean[pos] in FUNCTION_WORDS:
            return False
        # 量词后面不切 (All / new → 错，应该是 way / All)
        if clean[pos] in _pre_nominal:
            return False
        if tier == 1:
            # 后面紧跟紧密介词 (most of, one of, want to) → 不可分割
            if pos + 1 < n and clean[pos + 1] in _tight_after:
                return False
            # 后面紧跟宾语代词 (direct you, tell me, give him) → 动+宾不可切
            if pos + 1 < n and clean[pos + 1] in _pronoun_objects:
                return False
        return True

    def _find_boundary_in_range(lo, hi, search_range=None):
        """在 (lo, hi) 区间内找最佳切分点。
        先搜所有 tier 1 候选，选最接近中点的；没有则搜 tier 2。
        再找不到返回 None（调用方用中点兜底）。"""
        if search_range is None:
            search_range = max(5, (hi - lo) // 4)
        mid = lo + (hi - lo) // 2
        # 收集所有候选（offset, position, tier）
        candidates = []
        for offset in range(search_range + 1):
            for cand in (mid - offset, mid + offset):
                if _good_boundary(cand, lo, hi, tier=1):
                    candidates.append((abs(cand - mid), cand, 1))
                elif _good_boundary(cand, lo, hi, tier=2):
                    candidates.append((abs(cand - mid), cand, 2))
        if not candidates:
            return None
        # 优先 tier 1，其次距离中点最近
        candidates.sort(key=lambda x: (x[2], x[0]))
        return candidates[0][1]

    sorted_bounds = sorted(boundaries)
    last = -1
    for bi in sorted_bounds:
        gap_size = bi - last
        if gap_size > _long_gap_threshold:
            best = _find_boundary_in_range(last, bi)
            if best is None:
                best = last + gap_size // 2  # 最终兜底：中点
            boundaries.add(best)
        last = bi

    # 处理最后一段（尾段用更宽松的阈值）
    final_gap = (n - 1) - last
    if final_gap > _tail_threshold:
        best = _find_boundary_in_range(last, n - 1)
        if best is None:
            best = last + final_gap // 2
        boundaries.add(best)

    # 递归：新增边界后可能产生新的超长段，再扫一遍
    # （最多递归 2 次，避免无限循环）
    for _recursion in range(2):
        sorted_bounds = sorted(boundaries)
        last = -1
        added = False
        for bi in sorted_bounds:
            if bi - last > _long_gap_threshold:
                best = _find_boundary_in_range(last, bi)
                if best is not None:
                    boundaries.add(best)
                    added = True
            last = bi
        if (n - 1) - last > _tail_threshold:
            best = _find_boundary_in_range(last, n - 1)
            if best is not None:
                boundaries.add(best)
                added = True
        if not added:
            break

    return sorted(boundaries)


def _format_actual_pauses(words, gaps, suggested_boundaries=None):
    """
    学生实际停顿（中性标注，不加颜色判断）。
    / = 短停顿 120–300ms，// = 长停顿 >300ms。
    对照"建议意群"行即可自行判断停顿是否合理。
    """
    if not words:
        return ""
    gap_map = {i: g for i, g in enumerate(gaps)}
    parts = []
    for i, w in enumerate(words):
        parts.append(w["word"])
        if i < len(words) - 1:
            g = gap_map.get(i)
            if g and g["gap_ms"] >= 120:
                ms = g["gap_ms"]
                marker = " // " if ms >= 300 else " / "
                parts.append(marker)
            else:
                parts.append(" ")
    return "".join(parts)


def _format_suggested_groups(original_text, boundaries):
    """
    建议意群（蓝色 / 标记），基于语法结构。
    即使没有检测到边界，也返回原文（意味着"整句一口气读完"）。
    """
    if not original_text:
        return ""
    words = original_text.split()
    if not boundaries:
        # 无建议边界 → 返回原文，说明一句话到底
        return " ".join(words) + "  <span style='color:#888'>(整句一气呵成，无需切分)</span>"
    bset = set(boundaries)
    parts = []
    for i, w in enumerate(words):
        parts.append(w)
        if i in bset and i < len(words) - 1:
            parts.append(f' <span style="color:#1565C0;font-weight:bold">/</span> ')
        elif i < len(words) - 1:
            parts.append(" ")
    return "".join(parts)


def _compare_sense_groups(gaps, words, suggested_boundaries):
    """
    对比实际停顿 vs 建议意群，返回不匹配列表。
    """
    mismatches = []
    if not suggested_boundaries:
        return mismatches
    pause_map = {}
    for i, g in enumerate(gaps):
        pause_map[i] = g["gap_ms"]
    bset = set(suggested_boundaries)
    for i in range(len(words) - 1):
        is_boundary = i in bset
        gap_ms = pause_map.get(i, 0)
        has_pause = gap_ms >= 120
        w1 = words[i]["word"] if i < len(words) else "?"
        w2 = words[i + 1]["word"] if i + 1 < len(words) else "?"
        if is_boundary and not has_pause:
            mismatches.append({
                "type": "missing", "at": f"{w1} → {w2}",
                "desc": "⚠️ 该停没停", "gap_ms": gap_ms,
            })
        elif not is_boundary and has_pause and gap_ms >= 300:
            mismatches.append({
                "type": "extra_long", "at": f"{w1} → {w2}",
                "desc": "❌ 不该停却长停", "gap_ms": gap_ms,
            })
        elif not is_boundary and has_pause:
            mismatches.append({
                "type": "extra_short", "at": f"{w1} → {w2}",
                "desc": "🟡 不该停却短停", "gap_ms": gap_ms,
            })
    return mismatches


def analyze_rhythm(whisper_result, audio_duration, ground_truth_text=None, speech_onset=None):
    """分析节奏特征：词间间隙、虚词比例、蹦词检测

    speech_onset: 能量检测到的真实语音起点（秒）。用于修正首词 duration，
                  防止 Whisper 将句首静音计入首词时长 → 误判为"拖音"。"""
    words = whisper_result["words"]
    if len(words) < 3:
        return {"error": "Too few words"}

    # 有效词（有 whisper 时间戳的）用于统计基线
    valid_words = [w for w in words if w.get("duration") is not None]
    durations = [w["duration"] for w in valid_words]
    if not durations:
        return {"error": "No valid word timestamps"}
    dur_mean = np.mean(durations)
    dur_std = np.std(durations)

    # 建议意群边界（语法结构推断）
    # 优先用原文（ground truth），无原文时用 Whisper 识别文本作为 fallback。
    # 长文本（如 SGD）Whisper 常缺标点 → 先做标点恢复，让语法分析有更多锚点。
    _text_for_groups = None
    if ground_truth_text and ground_truth_text.strip():
        _text_for_groups, _ = _restore_punctuation(ground_truth_text.strip())
    elif whisper_result.get("text", "").strip():
        # 无原文兜底：用 Whisper 识别文本 + 标点恢复
        _text_for_groups, _ = _restore_punctuation(whisper_result["text"].strip())

    suggested_boundaries = _suggest_sense_groups(_text_for_groups) if _text_for_groups else []
    gt_words = _text_for_groups.split() if _text_for_groups else []
    gt_clean = [w.strip(".,;:!?\"'()[]{}—").lower() for w in gt_words]

    # ── 词间间隙 ──
    gaps = []
    for i in range(len(words) - 1):
        w_curr = words[i]
        w_next = words[i + 1]
        # 缺失词无法计算间隙
        if w_curr.get("end") is None or w_next.get("start") is None:
            continue
        gap_sec = w_next["start"] - w_curr["end"]
        w_before_raw = w_curr["word"]

        # 判断是否合法意群边界：语法建议 ∩（Whisper 标点检测作为补充）
        is_legit = (i in suggested_boundaries or
                    any(w_before_raw.rstrip().endswith(p) for p in [".", ",", ";", ":", "!", "?"]))

        gaps.append({
            "between": f"{w_curr['word']} → {w_next['word']}",
            "gap_sec": round(gap_sec, 3),
            "gap_ms": round(gap_sec * 1000),
            "is_sense_group_boundary": is_legit,
        })

    gap_values = [g["gap_sec"] for g in gaps]
    mean_gap = np.mean(gap_values) if gap_values else 0
    # 超过120ms且不在意群边界 = 问题间隙
    noticeable_gaps = [g for g in gaps if g["gap_sec"] > 0.12 and not g["is_sense_group_boundary"]]
    # 意群边界的正常停顿（不扣分）
    legit_pauses = [g for g in gaps if g["gap_sec"] > 0.12 and g["is_sense_group_boundary"]]
    # 超过200ms的非边界间隙
    large_gaps = [g for g in gaps if g["gap_sec"] > 0.20 and not g["is_sense_group_boundary"]]

    # ── 虚词 vs 实词时长比 ──
    func_words_in = []
    content_words_in = []
    for w in valid_words:
        # 优先用 aligned 的 clean 字段（来自原文），其次从 word 提取
        clean = w.get("clean") or w["word"].lower().strip(".,;:!?\"'")
        if clean in FUNCTION_WORDS:
            func_words_in.append(w)
        else:
            content_words_in.append(w)

    func_dur_mean = np.mean([w["duration"] for w in func_words_in]) if func_words_in else 0
    content_dur_mean = np.mean([w["duration"] for w in content_words_in]) if content_words_in else 0
    # 英语母语者：content/func ≈ 2.0+；中国学习者常 ≈ 1.0–1.5
    ratio = content_dur_mean / func_dur_mean if func_dur_mean > 0 else 0

    # ── 拖音检测（虚词 + 实词，相对语速自适应）──
    # 不用绝对值一刀切——慢语速的人所有词都长。
    # 用"比自己的平均时长长多少"来判断：
    #   虚词：超过自身均值的 60%（且至少 0.25s——太短无感知意义）
    #   实词：超过自身均值的 50%（且至少 0.40s）
    dragged_func = []
    dragged_content = []
    for i, w in enumerate(valid_words):
        clean_w = w.get("clean") or w["word"].lower().strip(".,;:!?\"'")
        is_func = clean_w in FUNCTION_WORDS

        # ── 首词时长修正 ──
        # Whisper 在句首有长静音时，可能将部分静音计入首词的 start 时间，
        # 导致 duration = end - start 虚高，把正常短虚词（如 "in"）误判为"拖音"。
        # 修正：用能量检测的真实语音起点替代 Whisper 偏早的 start。
        effective_dur = w["duration"]
        if i == 0 and speech_onset and speech_onset > 0.10:
            w_start = w.get("start")
            if w_start is not None and speech_onset > w_start + 0.03:
                # Whisper start 早于真实语音起点 → 静音被计入，需修正
                effective_dur = max(0.05, w.get("end", w_start + 0.1) - speech_onset)

        over_ratio = effective_dur / dur_mean if dur_mean > 0 else 1
        if is_func:
            if over_ratio > 1.6 and effective_dur > 0.25:
                dragged_func.append({
                    "word": w["word"],
                    "duration": round(effective_dur, 2),
                    "mean_dur": round(dur_mean, 2),
                    "over_ratio": round(over_ratio, 1),
                })
        else:
            if over_ratio > 1.5 and effective_dur > 0.40:
                dragged_content.append({
                    "word": w["word"],
                    "duration": round(effective_dur, 2),
                    "mean_dur": round(dur_mean, 2),
                    "over_ratio": round(over_ratio, 1),
                })

    # ── 蹦词检测 ──
    staccato_score = 0
    staccato_reasons = []

    # 有任意非边界的明显间隙就给基础分
    gap_count = len(noticeable_gaps)  # noticeable_gaps 已经排除了意群边界
    if gap_count > 0:
        staccato_score += min(gap_count, 3)
        staccato_reasons.append(f"{gap_count} 处非意群间隙 (>120ms)")

    # ── 意群分析（实际停顿 vs 建议切分）──
    actual_pauses_text = _format_actual_pauses(words, gaps, suggested_boundaries)
    if ground_truth_text:
        suggested_groups_text = _format_suggested_groups(ground_truth_text, suggested_boundaries)
    else:
        suggested_groups_text = ""  # 无原文 → 不瞎猜，报告里显示提示
    sense_group_comparison = _compare_sense_groups(gaps, words, suggested_boundaries)

    # ── 卡顿检测 ──
    # 填充词（有声卡顿）
    filler_words = {"um", "uh", "er", "ah", "mm", "hmm", "eh"}
    fillers_found = []
    for w in words:
        clean = w.get("clean") or w["word"].lower().strip(".,;:!?\"'")
        if clean in filler_words:
            fillers_found.append({"word": w["word"], "position": w.get("start")})

    # 长停顿（无声卡顿）：非标点处 >=500ms
    hesitation_gaps = [g for g in gaps if g["gap_sec"] >= 0.50 and not g["is_sense_group_boundary"]]
    hesitation_count = len(fillers_found) + len(hesitation_gaps)

    # 合法意群停顿仅作记录，不扣分
    if legit_pauses:
        pause_desc = ", ".join([g["between"] for g in legit_pauses[:3]])
        staccato_reasons.append(f"合法意群停顿: {pause_desc}")

    if mean_gap > 0.06:  # 降低阈值：60ms 以上就算有倾向
        staccato_score += 1
        staccato_reasons.append(f"平均间隙 {mean_gap*1000:.0f}ms 偏大")

    if ratio < 1.5 and ratio > 0:
        staccato_score += 1
        staccato_reasons.append(f"虚实词时长比仅 {ratio:.1f}:1（应 ≥1.6）")

    # 虚词拖音也是蹦词的伴生症状
    if len(dragged_func) > 0:
        staccato_score += 1
        staccato_reasons.append(f"{len(dragged_func)} 个虚词拖长 → 拖虚词想实词")
    if len(dragged_content) > 0:
        staccato_reasons.append(f"{len(dragged_content)} 个实词拖长 → 重读音节过度拉长")

    if dur_std / dur_mean < 0.22:  # 稍微提高阈值
        staccato_score += 1
        staccato_reasons.append("词长过于均匀，像等时节奏")

    # ── 语速 ──
    # 使用有效词的首尾时间戳（缺失词无时间戳）
    valid_with_time = [w for w in words if w.get("start") is not None and w.get("end") is not None]
    if valid_with_time:
        total_speech = valid_with_time[-1]["end"] - valid_with_time[0]["start"]
        wpm = len(words) / total_speech * 60 if total_speech > 0 else 0
    else:
        wpm = 0

    # ── 低置信度词（跳过缺失词的 None prob）──
    low_conf = [w for w in words if w.get("prob") is not None and w["prob"] < 0.5]
    very_low_conf = [w for w in words if w.get("prob") is not None and w["prob"] < 0.3]

    # ── 实词太短检测 ──
    too_short = [w for w in content_words_in if w["duration"] < dur_mean * 0.5 and w["duration"] < 0.2]

    return {
        "total_words": len(words),
        "wpm": round(wpm, 1),
        "wpm_verdict": "偏慢" if wpm < 110 else "正常" if wpm < 200 else "偏快",
        "actual_pauses_text": actual_pauses_text,
        "suggested_groups_text": suggested_groups_text,
        "sense_group_mismatches": sense_group_comparison,
        "suggested_boundaries": suggested_boundaries,
        "annotated_text": actual_pauses_text,  # 向后兼容旧字段名
        "audio_duration": round(audio_duration, 2),
        # 间隙
        "mean_gap_ms": round(mean_gap * 1000),
        "noticeable_gaps": noticeable_gaps,
        "legit_pauses": legit_pauses,
        "large_gaps": large_gaps,
        # 虚实词
        "func_word_count": len(func_words_in),
        "content_word_count": len(content_words_in),
        "func_dur_mean": round(func_dur_mean, 2),
        "content_dur_mean": round(content_dur_mean, 2),
        "content_func_ratio": round(ratio, 1),
        "ratio_verdict": "✓ 虚实词时长比例合理" if ratio > 1.6 else
                         "⚠️ 虚实词时长接近，蹦词/等时节奏特征" if ratio < 1.4 else
                         "🟡 虚实词比例偏低，有蹦词倾向",
        "dragged_func_words": dragged_func,       # 向后兼容
        "dragged_func": dragged_func,
        "dragged_content": dragged_content,
        # 蹦词
        "staccato_score": staccato_score,
        "staccato_verdict": "🔴 蹦词明显" if staccato_score >= 4 else
                           "🟡 有蹦词倾向" if staccato_score >= 1 else
                           "✓ 词间衔接正常",
        "staccato_reasons": staccato_reasons,
        # 发音
        "low_confidence_words": low_conf,
        "very_low_confidence_words": very_low_conf,
        "too_short_content_words": too_short,
        # 基础统计
        "dur_mean": round(dur_mean, 2),
        "dur_std": round(dur_std, 2),
        "dur_cv": round(dur_std / dur_mean, 2) if dur_mean > 0 else 0,
        # 卡顿
        "fillers_found": fillers_found,
        "hesitation_gaps": hesitation_gaps,
        "hesitation_count": hesitation_count,
        "hesitation_verdict": f"🔴 {hesitation_count} 处卡顿" if hesitation_count >= 3 else
                              f"🟡 {hesitation_count} 处卡顿" if hesitation_count >= 1 else
                              "✓ 未检测到卡顿",
    }


# ══════════════════════════════════════════════════
# 3.5 CMU 音素级分析 — 重音位置 + 发音偏差定位
# ══════════════════════════════════════════════════

def analyze_word_stress(words, whisper_result):
    """基于 CMU 发音词典 + 音频的音素级分析。"""
    stress_errors = []
    phoneme_deviations = []
    difficult_phoneme_words = []

    for w in words:
        clean = w.get("clean") or clean_word(w.get("word", ""))
        if not clean:
            continue
        is_func = clean in FUNCTION_WORDS
        match_status = w.get("match", "exact")

        # ── 3. 难点音素 → 仅在有实际发音证据时标记 ──
        # 证据 A：Whisper 置信度低 + 含难点音素 → "这些音可能没发准"
        # 证据 B：Whisper 识别替换 → 由 phoneme_deviations 处理（更精确）
        ph_list = _get_phonemes(clean)
        prob = w.get("prob")
        if ph_list and prob is not None and prob < 0.5 and not is_func:
            diff_ph = sorted(set(
                ph.rstrip("012") for ph in ph_list
                if ph.rstrip("012") in _CN_DIFFICULT_SET
            ))
            if diff_ph:
                difficult_phoneme_words.append({
                    "word": w["word"],
                    "clean": clean,
                    "phonemes": " ".join(ph_list),
                    "difficult": [f"{dp} → {CN_DIFFICULT_PHONEMES.get(dp, dp)}" for dp in diff_ph],
                    "prob": round(prob, 2),
                    "reason": "low_confidence",
                })

        # ── 1. 重音位置分析（3+音节实词）──
        if not is_func and match_status not in ("missing",):
            pattern = _get_stress_pattern(clean)
            if pattern and pattern["n_syllables"] >= 3:
                # CMU 词典参考信息，不做音频检测（等分音节法不可靠）
                stress_errors.append({
                    "word": w["word"],
                    "clean": clean,
                    "n_syllables": pattern["n_syllables"],
                    "expected_primary_syl": pattern["primary_syllable"] + 1,
                    "phoneme_sequence": " ".join(pattern["phoneme_sequence"]),
                    "verdict": "info",
                })

        # ── 2. 发音偏差定位（Whisper 替换/模糊匹配）──
        if match_status in ("substitution", "fuzzy"):
            whisper_w = clean_word(w.get("whisper_word") or "")
            if whisper_w and whisper_w != clean:
                diff = _phonemes_diff(clean, whisper_w)
                if diff and diff.get("is_minimal_pair"):
                    phoneme_deviations.append({
                        "word": w["word"],
                        "clean": clean,
                        "recognized_as": w.get("whisper_word", ""),
                        "phoneme_diff": diff,
                        "prob": w.get("prob"),
                    })

    return {
        "stress_errors": stress_errors,
        "phoneme_deviations": phoneme_deviations,
        "difficult_phoneme_words": difficult_phoneme_words,
    }


# ══════════════════════════════════════════════════
# 4. 序列对齐（处理 Whisper 识别偏差）
# ══════════════════════════════════════════════════

def clean_word(w):
    """清洗词：去标点、小写"""
    return w.lower().strip(".,;:!?\"'()[]{}")

def split_text(text):
    """把文本拆成清洗后的词列表"""
    return [clean_word(w) for w in text.split()]

def align_whisper_to_original(whisper_words, original_text):
    """
    将 Whisper 识别结果对齐到原文，返回统一格式的 aligned words。

    每项:
      word          — 原文词（显示用）
      whisper_word  — Whisper 实际识别到的词
      clean          — 清理后的原文词（虚词判断/匹配用）
      start/end/duration/prob — 来自 whisper（缺失=None）
      match          — "exact" / "fuzzy" / "substitution" / "missing" / "extra"

    同时返回 stats: {matched, substitution, missing, extra}
    """
    raw_words = original_text.split()
    gt_clean = [clean_word(w) for w in raw_words]
    whisper_clean = [clean_word(w["word"]) for w in whisper_words]

    # ── Step 1: SequenceMatcher 精确匹配 ──
    sm = SequenceMatcher(None, whisper_clean, gt_clean)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]

    # 初始化：按原文顺序
    result = []
    for gi, raw_w in enumerate(raw_words):
        result.append({
            "word": raw_w,                     # 原文词（保留标点，显示用）
            "whisper_word": None,
            "clean": gt_clean[gi],
            "start": None,
            "end": None,
            "duration": None,
            "prob": None,
            "match": "missing",
        })

    matched_detected = set()

    for block in blocks:
        for k in range(block.size):
            di = block.a + k   # whisper index
            gi = block.b + k   # gt index
            w = whisper_words[di]
            result[gi].update({
                "whisper_word": w["word"],
                "start": w["start"],
                "end": w["end"],
                "duration": w["duration"],
                "prob": w.get("prob", 1.0),
                "match": "exact" if whisper_clean[di] == gt_clean[gi] else "substitution",
            })
            matched_detected.add(di)

    # ── Step 2: 模糊匹配剩余词（处理 Whisper 小偏差） ──
    unmatched_gt = [i for i, r in enumerate(result) if r["match"] == "missing"]
    unmatched_det = [i for i in range(len(whisper_words)) if i not in matched_detected]

    for di in unmatched_det:
        dw = whisper_clean[di]
        best_ratio = 0
        best_gi = None
        for gi in unmatched_gt:
            gw = gt_clean[gi]
            ratio = SequenceMatcher(None, dw, gw).ratio()
            if ratio > best_ratio and ratio > 0.5:
                best_ratio = ratio
                best_gi = gi
        if best_gi is not None:
            w = whisper_words[di]
            is_exact = whisper_clean[di] == gt_clean[best_gi]
            result[best_gi].update({
                "whisper_word": w["word"],
                "start": w["start"],
                "end": w["end"],
                "duration": w["duration"],
                "prob": w.get("prob", 1.0),
                "match": "exact" if is_exact else "fuzzy",
            })
            unmatched_gt.remove(best_gi)
            matched_detected.add(di)

    # ── Step 3: 未匹配的 whisper 词 = extra（学生多读了） ──
    for di in range(len(whisper_words)):
        if di not in matched_detected:
            w = whisper_words[di]
            result.append({
                "word": w["word"],
                "whisper_word": w["word"],
                "clean": clean_word(w["word"]),
                "start": w["start"],
                "end": w["end"],
                "duration": w["duration"],
                "prob": w.get("prob", 1.0),
                "match": "extra",
            })

    # ── 统计 ──
    # "fuzzy" + "substitution" 都是发现词不一致（发音偏差）
    stats = {
        "matched": sum(1 for r in result if r["match"] == "exact"),
        "substitution": sum(1 for r in result if r["match"] in ("substitution", "fuzzy")),
        "missing": sum(1 for r in result if r["match"] == "missing"),
        "extra": sum(1 for r in result if r["match"] == "extra"),
        "total_original": len(raw_words),
        "total_spoken": len(whisper_words),
    }

    return {"words": result, "stats": stats}


# Backward-compatible wrapper for compare_with_native
def align_to_ground_truth(whisper_words, ground_truth_text):
    """旧接口：返回旧格式列表（供 compare_with_native 使用）"""
    aligned = align_whisper_to_original(whisper_words, ground_truth_text)
    return [{
        "gt_word": r["clean"],
        "whisper_word": r["whisper_word"],
        "duration": r["duration"],
        "prob": r["prob"],
        "start": r["start"],
        "end": r["end"],
        "alignment": "matched" if r["match"] == "exact" else
                     "fuzzy_matched" if r["match"] == "fuzzy" else
                     "substitution" if r["match"] == "substitution" else
                     "missing_in_whisper",
    } for r in aligned["words"] if r["match"] != "extra"]


def _wrap_whisper_as_aligned(whisper_words):
    """无原文时：把 whisper 原词包装成 aligned 格式，兼容下游函数。"""
    return [{
        "word": w["word"],
        "whisper_word": w["word"],
        "clean": clean_word(w["word"]),
        "start": w["start"],
        "end": w["end"],
        "duration": w["duration"],
        "prob": w.get("prob", 1.0),
        "match": "exact",
    } for w in whisper_words]


# ══════════════════════════════════════════════════
# 5. 母语者对比（支持 ground truth 对齐）
# ══════════════════════════════════════════════════

def compare_with_native(student_whisper, native_whisper, student_pitch, native_pitch,
                         ground_truth_text=None):
    """
    对比学生 vs 母语者。
    如果提供 ground_truth_text，先对齐到原文再做比较，
    避免 Whisper 识别偏差导致错位对比。
    """
    # 如果提供了 ground truth，先对齐
    if ground_truth_text:
        gt_words = split_text(ground_truth_text)
        stud_aligned = align_to_ground_truth(student_whisper["words"], ground_truth_text)
        native_aligned = align_to_ground_truth(native_whisper["words"], ground_truth_text)

        # 为每个 gt word 对齐学生和母语者
        comparisons = []
        alignment_stats = {"matched": 0, "student_missing": 0, "native_missing": 0, "both_missing": 0}

        for i, gt_word in enumerate(gt_words):
            s = stud_aligned[i] if i < len(stud_aligned) else None
            n = native_aligned[i] if i < len(native_aligned) else None

            s_dur = s["duration"] if s and s["duration"] is not None else None
            n_dur = n["duration"] if n and n["duration"] is not None else None
            s_prob = s["prob"] if s and s["prob"] is not None else None

            if s_dur is None and n_dur is None:
                alignment_stats["both_missing"] += 1
                # 检查是否有模糊匹配（学生发音偏离但被识别到了）
                s_heard = s.get("whisper_word") if s else None
                n_heard = n.get("whisper_word") if n else None
                s_align = s.get("alignment") if s else None
                n_align = n.get("alignment") if n else None

                if s_align == "fuzzy_matched":
                    verdict = f"🔴 发音偏离: 语音模型识别为'{s_heard}'而非'{gt_word}'"
                elif s_heard:
                    verdict = f"⚠️ 学生发音存疑: '{s_heard}'≠'{gt_word}'"
                else:
                    verdict = "⚠️ 双方未对齐"

                comparisons.append({
                    "word": gt_word,
                    "student_dur": round(s_dur, 2) if s_dur else None,
                    "student_heard_as": s_heard,
                    "native_dur": round(n_dur, 2) if n_dur else None,
                    "native_heard_as": n_heard,
                    "diff_ms": None,
                    "ratio": None,
                    "verdict": verdict,
                    "student_prob": s_prob,
                    "student_alignment": s_align,
                })
                continue
            elif s_dur is None:
                alignment_stats["student_missing"] += 1
                comparisons.append({
                    "word": gt_word,
                    "student_dur": None,
                    "native_dur": round(n_dur, 2),
                    "diff_ms": None,
                    "ratio": None,
                    "verdict": "⚠️ 学生漏词/未识别",
                    "student_prob": None,
                })
                continue
            elif n_dur is None:
                alignment_stats["native_missing"] += 1
                comparisons.append({
                    "word": gt_word,
                    "student_dur": round(s_dur, 2),
                    "native_dur": None,
                    "diff_ms": None,
                    "ratio": None,
                    "verdict": "⚠️ 母语者漏词/未识别",
                    "student_prob": s_prob,
                })
                continue

            alignment_stats["matched"] += 1
            dur_diff = s_dur - n_dur
            dur_ratio = s_dur / n_dur if n_dur > 0 else 1

            verdict = ""
            if dur_ratio > 1.8:
                verdict = "🔴 明显拖长"
            elif dur_ratio > 1.4:
                verdict = "🟡 偏长"
            elif dur_ratio < 0.5:
                verdict = "⚠️ 太短促"
            elif dur_ratio < 0.7:
                verdict = "🟡 偏短"
            else:
                verdict = "✓"

            comparisons.append({
                "word": gt_word,
                "student_dur": round(s_dur, 2),
                "native_dur": round(n_dur, 2),
                "diff_ms": round(dur_diff * 1000),
                "ratio": round(dur_ratio, 1),
                "verdict": verdict,
                "student_prob": s_prob,
            })

        # 整体统计（仅匹配到的词）
        matched_comps = [c for c in comparisons if c["student_dur"] is not None and c["native_dur"] is not None]
        stud_total = sum(c["student_dur"] for c in matched_comps)
        native_total = sum(c["native_dur"] for c in matched_comps)
        overall_ratio = stud_total / native_total if native_total > 0 else 1

        return {
            "word_comparisons": comparisons,
            "alignment_stats": alignment_stats,
            "student_total_dur": round(stud_total, 2),
            "native_total_dur": round(native_total, 2),
            "overall_dur_ratio": round(overall_ratio, 2),
            "pitch_comparison": _pitch_compare(student_pitch, native_pitch),
        }

    # 没有 ground truth — 回退到逐位置对比
    stud_words = student_whisper["words"]
    native_words = native_whisper["words"]

    comparisons = []
    n = min(len(stud_words), len(native_words))

    for i in range(n):
        sw = stud_words[i]
        nw = native_words[i]
        dur_diff = sw["duration"] - nw["duration"]
        dur_ratio = sw["duration"] / nw["duration"] if nw["duration"] > 0 else 1

        verdict = ""
        if dur_ratio > 1.8:
            verdict = "🔴 明显拖长"
        elif dur_ratio > 1.4:
            verdict = "🟡 偏长"
        elif dur_ratio < 0.5:
            verdict = "⚠️ 太短促"
        elif dur_ratio < 0.7:
            verdict = "🟡 偏短"
        else:
            verdict = "✓"

        comparisons.append({
            "word": sw["word"],
            "student_dur": round(sw["duration"], 2),
            "native_dur": round(nw["duration"], 2),
            "diff_ms": round(dur_diff * 1000),
            "ratio": round(dur_ratio, 1),
            "verdict": verdict,
            "student_prob": sw.get("prob", 1.0),
        })

    stud_total = sum(w["duration"] for w in stud_words)
    native_total = sum(w["duration"] for w in native_words)
    overall_ratio = stud_total / native_total if native_total > 0 else 1

    return {
        "word_comparisons": comparisons,
        "alignment_stats": None,
        "student_total_dur": round(stud_total, 2),
        "native_total_dur": round(native_total, 2),
        "overall_dur_ratio": round(overall_ratio, 2),
        "pitch_comparison": _pitch_compare(student_pitch, native_pitch),
    }


def _pitch_compare(student_pitch, native_pitch):
    """提取音高对比"""
    if student_pitch.get("mean_f0", 0) > 0 and native_pitch.get("mean_f0", 0) > 0:
        return {
            "student_mean_f0": student_pitch["mean_f0"],
            "native_mean_f0": native_pitch["mean_f0"],
            "student_range": student_pitch["f0_range"],
            "native_range": native_pitch["f0_range"],
            "student_variation": student_pitch["variation"],
            "native_variation": native_pitch.get("variation", "?"),
        }
    return {}


# ══════════════════════════════════════════════════
# 5. 逐词诊断（精确到每个词的错误类型）
# ══════════════════════════════════════════════════

def _word_pitch_features(f0, times, word_start, word_end):
    """提取单个词时间窗口内的音高特征"""
    mask = (times >= word_start) & (times <= word_end)
    word_f0 = f0[mask]
    word_f0_voiced = word_f0[word_f0 > 0]

    if len(word_f0_voiced) < 3:
        return {"has_pitch": False, "mean_f0": None, "slope": None, "direction": "无数据"}

    mean_f0 = float(np.mean(word_f0_voiced))
    # 词内音高走势：线性回归斜率
    x = np.arange(len(word_f0_voiced))
    if len(x) > 1:
        slope = float(np.polyfit(x, word_f0_voiced, 1)[0])  # Hz/帧
    else:
        slope = 0.0

    # 方向判断
    if slope > 3:
        direction = "↑ 上扬"
    elif slope < -3:
        direction = "↓ 下降"
    else:
        direction = "→ 平稳"

    return {
        "has_pitch": True,
        "mean_f0": round(mean_f0, 1),
        "slope": round(slope, 1),
        "direction": direction,
        "n_frames": len(word_f0_voiced),
    }


def diagnose_per_word(whisper_result, rhythm, comparison=None, pitch=None, f0_array=None, f0_times=None, speech_onset=None):
    """
    对每个词做精确诊断，标记具体错误类型。
    错误标签体系（来自错误类型库）：
      🔴 严重偏离    — Whisper 置信度 < 0.3，可能吞音/含糊/完全读错
      🟡 元音不饱满   — 中低置信度实词，可能是元音/双元音没张开
      🟡 辅音不准    — 中低置信度词尾辅音，可能漏s/加呃/变音
      🟡 拖音       — 时长明显超过参考
      🟡 虚词拖长    — 虚词时长过长（在用它想下一个词）
      🟡 实词太短    — 实词时长明显短于参考
      ⚠️ 蹦词间隙    — 该词后有明显间隙（词与词脱节）
      ⚠️ 句首太轻    — 句首虚词/代词过短
      ✓ 正常
    """
    words = whisper_result["words"]
    word_diag = []

    # ── 整体清晰度：匹配率高+置信度高 → 录音清晰，抑制轻量警告 ──
    alignment_stats = whisper_result.get("alignment_stats")
    probs_ok = [w.get("prob") for w in words if w.get("prob") is not None]
    avg_prob = np.mean(probs_ok) if probs_ok else 0
    clean_audio = False
    if alignment_stats:
        total = alignment_stats["total_original"]
        good = alignment_stats["matched"]
        bad = alignment_stats["substitution"] + alignment_stats["missing"]
        clean_audio = (total > 0 and good / total >= 0.85 and bad <= 1 and avg_prob > 0.8)
    else:
        clean_audio = (probs_ok and all(p >= 0.7 for p in probs_ok))

    # 基础统计数据（仅有效词）
    valid_durs = [w["duration"] for w in words if w.get("duration") is not None]
    dur_mean = np.mean(valid_durs) if valid_durs else 0.3

    # 间隙数据（区分问题间隙 vs 合法意群停顿）
    all_gaps = rhythm.get("noticeable_gaps", []) + rhythm.get("legit_pauses", [])
    gap_after = {}  # word_index → {"gap_ms": int, "is_legit": bool}
    for g in all_gaps:
        parts = g["between"].split(" → ")
        if len(parts) == 2:
            for i, w in enumerate(words):
                if i < len(words) - 1 and w["word"] == parts[0] and words[i+1]["word"] == parts[1]:
                    gap_after[i] = {
                        "gap_ms": g["gap_ms"],
                        "is_legit": g.get("is_sense_group_boundary", False),
                    }
                    break

    # 参考对比数据
    ref_durs = {}  # word → native_duration
    if comparison and comparison.get("word_comparisons"):
        for c in comparison["word_comparisons"]:
            key = clean_word(c["word"])
            if c.get("native_dur") is not None:
                ref_durs[key] = c["native_dur"]

    for i, w in enumerate(words):
        word = w["word"]
        clean = w.get("clean") or clean_word(word)
        match_status = w.get("match", "exact")
        prob = w.get("prob")  # may be None for missing words
        dur = w.get("duration")  # may be None for missing words
        is_func = clean in FUNCTION_WORDS
        is_last = (i == len(words) - 1)
        is_first = (i == 0)
        ref_dur = None  # 初始化为 None，缺失词不会进入时长对比

        issues = []
        severity = "ok"  # ok / warning / error

        # ── 对齐状态：缺失/替换词（优先检测）──
        if match_status == "missing":
            issues.append("🔴 未读/吞词 — 原文有这个单词但录音中未检测到")
            severity = "error"
        elif match_status in ("substitution", "fuzzy"):
            whisper_w = w.get("whisper_word", "")
            issues.append(f"🔴 发音偏差 → 语音模型识别为 '{whisper_w}'，与原文 '{word}' 不匹配")
            severity = "error"
        elif match_status == "extra":
            issues.append("⚠️ 多读/插入词 — 原文中不存在此词")
            if severity == "ok":
                severity = "warning"

        # ── 置信度分析（仅有效词）──
        if prob is not None and match_status not in ("missing",):
            if prob < 0.3:
                issues.append("🔴 严重偏离：发音严重不清晰，可能吞音/含糊/读错词")
                severity = "error"
            elif prob < 0.5 and not clean_audio:
                if is_last:
                    issues.append("🟡 句尾词不清晰：可能尾音吞掉或拖音导致模糊")
                elif is_func:
                    # 虚词弱读置信度低是正常的，但过低也可能真有发音问题
                    issues.append("🟡 虚词发音模糊：可能未正确弱读或元音偏离")
                else:
                    if len(clean) > 5:
                        issues.append("🟡 多音节词不清晰：可能音节松散/重音不对/元音不饱满")
                    else:
                        issues.append("🟡 元音不饱满或辅音不准")
                if severity != "error":
                    severity = "warning"
            elif prob < 0.7 and not is_func and not clean_audio:
                # 实词置信度偏低（0.5–0.7）：无论长短、是否句首，都标
                if len(clean) > 5:
                    issues.append("🟡 多音节词偏模糊：个别音节或重音可能不准")
                else:
                    issues.append("🟡 发音偏模糊：可能个别元音或辅音不够清晰")
                if severity == "ok":
                    severity = "warning"

        # ── 时长对比（有参考时，仅有效词）──
        if dur is not None:
            ref_dur = ref_durs.get(clean)
            if ref_dur and ref_dur > 0:
                ratio = dur / ref_dur
                if is_func and ratio > 1.6:
                    issues.append(f"🟡 虚词拖长 ({ratio:.1f}x参考) → 可能在用虚词想下一个词")
                    if severity == "ok":
                        severity = "warning"
                elif not is_func and ratio > 1.8:
                    issues.append(f"🔴 明显拖音 ({ratio:.1f}x参考) → 元音拉得过长")
                    severity = "error"
                elif not is_func and ratio > 1.4:
                    issues.append(f"🟡 偏长 ({ratio:.1f}x参考)")
                    if severity == "ok":
                        severity = "warning"
                elif not is_func and ratio > 1.25 and len(clean) > 6:
                    # 多音节词略长但未到偏长阈值 → 音节松散
                    issues.append(f"🟡 音节松散 ({ratio:.1f}x参考) → 多音节词内部不够紧凑")
                    if severity == "ok":
                        severity = "warning"
                elif not is_func and ratio < 0.5:
                    issues.append(f"⚠️ 实词太短促 ({ratio:.1f}x参考) → 实词没给够时长")
                    if severity == "ok":
                        severity = "warning"
                elif is_func and ratio < 0.4:
                    issues.append(f"⚠️ 虚词过短 ({ratio:.1f}x参考) → 可能吞掉了")
                    if severity == "ok":
                        severity = "warning"

            # ── 绝对时长判断（无参考时）──
            if not ref_dur:
                # 首词时长修正：Whisper 在句首有长静音时可能把静音计入 start，
                # 导致 duration 虚高。用能量检测的真实语音起点做修正。
                _effective_dur = dur
                if is_first and speech_onset and speech_onset > 0.10:
                    w_start = w.get("start")
                    if w_start is not None and dur is not None and speech_onset > w_start + 0.03:
                        w_end = w.get("end", w_start + 0.1)
                        _effective_dur = max(0.05, w_end - speech_onset)

                if is_func and _effective_dur > 0.35:
                    issues.append("🟡 虚词偏长 → 可能在用它缓冲/想词")
                    if severity == "ok":
                        severity = "warning"
                if not is_func and dur < 0.15:
                    issues.append("⚠️ 实词太短促 → 实词没给够发音时间")
                    if severity == "ok":
                        severity = "warning"
                # 用音节数归一化：5音节词自然比单音节词长
                nsyl = _get_stress_pattern(clean)
                nsyl_count = nsyl["n_syllables"] if nsyl else 1
                dur_per_syl = dur / nsyl_count if nsyl_count > 0 else dur
                # 每音节 > 0.25s 才算真正松散（任何语速下都偏慢）
                if not is_func and nsyl_count >= 2 and dur_per_syl > 0.25:
                    issues.append("🟡 音节松散 → 多音节词内部不够紧凑，音节之间拖得太开")
                    if severity == "ok":
                        severity = "warning"
                if is_last and dur > dur_mean * 1.5 and dur > 0.5:
                    issues.append("🟡 句尾拖音 → 最后一个词拉得太长")
                    if severity == "ok":
                        severity = "warning"

        # ── 间隙分析 ──
        gap_info = gap_after.get(i)
        if gap_info:
            gap_ms = gap_info["gap_ms"]
            is_legit = gap_info["is_legit"]

            if is_legit:
                pass
            else:
                if gap_ms >= 500:
                    issues.append(f"🔴 卡顿 ({gap_ms}ms) → 不该停的地方停了半秒以上")
                    severity = "error"
                elif gap_ms > 200:
                    if is_func:
                        issues.append(f"🔴 虚词后停顿 ({gap_ms}ms) → 在虚词后断了！")
                        severity = "error"
                    else:
                        issues.append(f"🟡 词后间隙 ({gap_ms}ms) → 蹦词/意群断裂")
                        if severity == "ok":
                            severity = "warning"
                elif gap_ms > 120:
                    if is_func:
                        issues.append(f"🟡 虚词后小停顿 ({gap_ms}ms)")
                        if severity == "ok":
                            severity = "warning"

        # ── 卡顿：填充词检测 ──
        if clean in {"um", "uh", "er", "ah", "mm", "hmm", "eh"}:
            issues.append("🔴 有声卡顿（填充词）→ 呃/嗯 类声音")
            severity = "error"

        # ── 词级音高分析（所有有效词，包括虚词）──
        if (f0_array is not None and f0_times is not None
                and w.get("start") is not None and w.get("end") is not None
                and match_status != "missing"):
            pf = _word_pitch_features(f0_array, f0_times, w["start"], w["end"])
            if pf["has_pitch"] and pitch:
                sent_mean = pitch.get("mean_f0", 150)
                # 音高明显偏离句子均值
                if pf["mean_f0"] and abs(pf["mean_f0"] - sent_mean) > 40 and not clean_audio:
                    if pf["mean_f0"] > sent_mean:
                        if is_func:
                            issues.append(f"🟡 虚词音高偏高 ({pf['mean_f0']:.0f}Hz) → 虚词不该重读")
                        else:
                            issues.append(f"🟡 音高偏高 ({pf['mean_f0']:.0f}Hz, 句子均值{sent_mean:.0f}Hz) → 可能重音过重或语调异常")
                    else:
                        if not is_func:
                            issues.append(f"🟡 音高偏低 ({pf['mean_f0']:.0f}Hz, 句子均值{sent_mean:.0f}Hz) → 可能该强调的词没给够音高")
                    if severity == "ok":
                        severity = "warning"
                # 陈述句中单个实词异常上扬
                if pf["direction"] == "↑ 上扬" and pf["slope"] > 5:
                    tail_ok = pitch.get("tail_verdict", "").startswith("✓") if pitch else False
                    if tail_ok and not is_last:
                        issues.append(f"⚠️ 词内异常上扬 ({pf['slope']:.0f}Hz/帧) → 陈述句中不应突然拔高")
                        if severity == "ok":
                            severity = "warning"

        # ── 句首判断 ──
        # 只有句首实词太短才是问题；句首虚词（the/a/it等）本来就该轻短
        if is_first and not is_func and dur is not None and dur < 0.25:
            issues.append("⚠️ 句首实词太短 → 开头没有给足分量")
            if severity == "ok":
                severity = "warning"

        # ── 汇总 ──
        if not issues:
            issues.append("✓ 正常")

        word_diag.append({
            "word": word,
            "clean": clean,
            "index": i,
            "match": match_status,
            "whisper_word": w.get("whisper_word"),
            "duration": round(dur, 2) if dur is not None else None,
            "prob": round(prob, 2) if prob is not None else None,
            "is_function_word": is_func,
            "ref_duration": round(ref_dur, 2) if ref_dur else None,
            "issues": issues,
            "severity": severity,  # ok / warning / error
        })

    return word_diag


def _weave_cmu_into_word_diag(word_diag, cmu_analysis, words):
    """将 CMU 音素级分析结果织入逐词诊断：重音错误、发音偏差、难点音素。"""
    if not cmu_analysis:
        return

    # 建立 index → stress_error / phoneme_deviation 的映射
    stress_map = {}
    for se in cmu_analysis.get("stress_errors", []):
        clean = se.get("clean", "")
        for wd in word_diag:
            if wd["clean"] == clean and wd.get("match") not in ("missing",):
                stress_map[wd["index"]] = se
                break

    phoneme_map = {}
    for pd_item in cmu_analysis.get("phoneme_deviations", []):
        clean = pd_item.get("clean", "")
        for wd in word_diag:
            if wd["clean"] == clean and wd.get("match") in ("substitution", "fuzzy"):
                phoneme_map[wd["index"]] = pd_item
                break

    diff_map = {}
    for dw in cmu_analysis.get("difficult_phoneme_words", []):
        clean = dw.get("clean", "")
        for wd in word_diag:
            if wd["clean"] == clean:
                diff_map[wd["index"]] = dw
                break

    # ── 构建 stress_map（仅词典信息，无音频检测）──
    stress_map = {}
    for se in cmu_analysis.get("stress_errors", []):
        clean = se.get("clean", "")
        for wd in word_diag:
            if wd["clean"] == clean and wd.get("match") not in ("missing",):
                stress_map[wd["index"]] = se
                break

    # 织入
    for wd in word_diag:
        idx = wd["index"]

        # 发音偏差定位（Whisper 替换 + CMU 对比）
        pd_item = phoneme_map.get(idx)
        if pd_item:
            diff = pd_item.get("phoneme_diff")
            if diff and diff.get("is_minimal_pair"):
                diffs = diff.get("diff_positions", [])
                for pos, t_ph, r_ph in diffs:
                    t_clean = t_ph.rstrip("012")
                    r_clean = r_ph.rstrip("012") if r_ph not in ("（省略）", "（插入）") else r_ph
                    if t_clean in _CN_DIFFICULT_SET:
                        detail = CN_DIFFICULT_PHONEMES.get(t_clean, "").split("—")[0].strip()
                        issue = (f"🔴 音素偏差：/ {t_clean} / → 语音模型识别为 / {r_clean} / "
                                 f"({detail})")
                    else:
                        issue = (f"🟡 音素偏差：目标 /{t_clean}/ → 实际近似 /{r_clean}/")
                    if not any(issue in ex for ex in wd["issues"]):
                        wd["issues"].append(issue)
                        if wd["severity"] == "ok":
                            wd["severity"] = "warning"

        # 难点音素 → 仅低置信度词的针对性标记
        dw = diff_map.get(idx)
        if dw and dw.get("difficult") and dw.get("reason") == "low_confidence":
            ph_tags = ", ".join(dw["difficult"][:2])
            issue = f"🟡 疑错音素（置信度{dw.get('prob', '?')}）：{ph_tags}"
            if not any("疑错音素" in ex for ex in wd["issues"]):
                wd["issues"].append(issue)
                if wd["severity"] == "ok":
                    wd["severity"] = "warning"

        # 重音位置（CMU 词典参考，不做音频检测）
        se = stress_map.get(idx)
        if se and not wd.get("is_function_word"):
            exp = se.get("expected_primary_syl", "?")
            nsyl = se.get("n_syllables", "?")
            info = f"💡 重音参考：{nsyl}音节词，主重音在第{exp}音节"
            if not any("重音参考" in ex for ex in wd["issues"]):
                wd["issues"].append(info)
    """将逐词诊断转为 Markdown 表格"""
    lines = []
    lines.append("| # | 词 | 时长 | 置信度 | 状态 | 类型 | 具体问题 |")
    lines.append("|---|-----|------|--------|------|------|---------|")

    for w in word_diag:
        ref_str = f"{w['ref_duration']}s" if w['ref_duration'] else "—"
        word_type = "虚词" if w['is_function_word'] else "实词"
        dur_str = f"{w['duration']}s" if w['duration'] is not None else "—"
        prob_str = f"{w['prob']:.0%}" if w['prob'] is not None else "—"

        if w['severity'] == 'error':
            prefix = '🔴 '
        elif w['severity'] == 'warning':
            prefix = '🟡 '
        else:
            prefix = ''

        # 对齐状态标记
        match_status = w.get("match", "")
        if match_status == "missing":
            match_label = "🔴 缺失"
        elif match_status in ("substitution", "fuzzy"):
            match_label = "🔴 替换"
        elif match_status == "extra":
            match_label = "⚠️ 多读"
        elif match_status in ("exact", "fuzzy"):
            match_label = "✓"
        else:
            match_label = "—"

        # 把所有问题拼在一起（去重、去掉✓正常）
        issue_list = [i for i in w['issues'] if i != "✓ 正常"]
        if not issue_list:
            issue_list = ["✓ 正常"]

        # 第一个问题带词名，后续问题另起行缩进
        issue_text = "<br>".join(issue_list)

        # 实词问题加发音（字典 + 学生，均内嵌；缺失词无学生原音）
        word_display = w['word']
        if w.get("whisper_word") and w["whisper_word"] != w["word"]:
            word_display += f' → <span style="color:#e74c3c">{w["whisper_word"]}</span>'
        if w['severity'] in ('error', 'warning') and not w.get('is_function_word', False):
            if w.get("dict_audio_uri"):
                word_display += f' 📖<audio controls style="height:1.2em;width:80px;display:inline-block;vertical-align:middle" src="{w["dict_audio_uri"]}" title="字典发音" preload="none"></audio>'
            if w.get("student_audio_uri"):
                word_display += f' 🎤<audio controls style="height:1.2em;width:80px;display:inline-block;vertical-align:middle" src="{w["student_audio_uri"]}" title="学生发音"></audio>'

        lines.append(
            f"| {w['index']+1} | {prefix}{word_display} | {dur_str} "
            f"| {prob_str} | {match_label} | {word_type} | {issue_text} |"
        )

    # 汇总所有非正常的词
    lines.append("")
    lines.append("### 需要关注的词")
    problem_words = [w for w in word_diag if w['severity'] != 'ok']
    if problem_words:
        for w in problem_words:
            for issue in w['issues']:
                if issue != "✓ 正常":
                    prob_str = f"{w['prob']:.0%}" if w.get('prob') is not None else "—"
                    dur_str = f"{w['duration']}s" if w.get('duration') is not None else "—"
                    lines.append(f"- {issue}: **{w['word']}** (时长 {dur_str}, 置信度 {prob_str})")
    else:
        lines.append("✓ 所有词在正常范围内")

    return "\n".join(lines)


# ══════════════════════════════════════════════════
# 6. 错误类型映射（按诊断模板）
# ══════════════════════════════════════════════════

def map_to_diagnostic_template(rhythm, pitch, whisper_result, comparison=None, cmu_analysis=None):
    """将分析结果映射到诊断模板的勾选项"""
    flags = {
        "RA.节奏": [],
        "RA.流利度": [],
        "RA.发音": [],
        "RA.策略": [],
    }

    # ── 整体清晰度判断 ──
    # 如果 Whisper 匹配率高且概率高，说明录音清晰、发音好，抑制轻量警告
    alignment_stats = whisper_result.get("alignment_stats")
    words = whisper_result.get("words", [])
    clean_audio = False
    if alignment_stats:
        total = alignment_stats["total_original"]
        good = alignment_stats["matched"]
        bad = alignment_stats["substitution"] + alignment_stats["missing"]
        clean_audio = (total > 0 and good / total >= 0.85 and bad <= 1)
    # 无原文时退而求其次：所有词置信度 ≥ 0.7
    if not alignment_stats and words:
        probs = [w.get("prob") for w in words if w.get("prob") is not None]
        if probs and all(p >= 0.7 for p in probs):
            clean_audio = True

    # ── 节奏 ──
    if rhythm["staccato_score"] >= 4:
        flags["RA.节奏"].append("🔴 蹦词：词与词之间有缝隙，每个词一样长一样重")
    elif rhythm["staccato_score"] >= 2:
        flags["RA.节奏"].append("🟡 蹦词倾向：部分词之间有可见缝隙")

    if rhythm.get("dragged_func"):
        examples = ", ".join([f"'{d['word']}'({d['duration']}s)" for d in rhythm["dragged_func"][:4]])
        flags["RA.节奏"].append(f"🔴 虚词拖音：{examples}")
    if rhythm.get("dragged_content") and not clean_audio:
        examples = ", ".join([f"'{d['word']}'({d['duration']}s)" for d in rhythm["dragged_content"][:4]])
        flags["RA.节奏"].append(f"🟡 实词拖音：{examples}")

    if rhythm.get("content_func_ratio", 2) < 1.3:
        flags["RA.节奏"].append("🟡 等时节奏：虚实词时长接近，中文节奏特征")

    if rhythm["dur_cv"] < 0.25:
        flags["RA.节奏"].append("🟡 词长过于均匀 → 可能意群不紧凑或缺少重弱读")

    if rhythm["too_short_content_words"]:
        examples = ", ".join([f"'{w['word']}'({w['duration']:.1f}s)" for w in rhythm["too_short_content_words"][:3]])
        flags["RA.节奏"].append(f"🟡 实词太短促：{examples}")

    # ── 卡顿 ──
    if rhythm.get("hesitation_count", 0) > 0:
        flags["RA.流利度"].append(rhythm.get("hesitation_verdict", ""))
        for g in rhythm.get("hesitation_gaps", [])[:3]:
            flags["RA.流利度"].append(f"无声卡顿: {g['between']} ({g['gap_ms']}ms)")
        for f in rhythm.get("fillers_found", [])[:3]:
            flags["RA.流利度"].append(f"有声卡顿: '{f['word']}' 在 {f['position']:.1f}s 处")

    # ── 流利度 ──
    if rhythm["large_gaps"]:
        gap_descs = [f"{g['between']}({g['gap_ms']}ms)" for g in rhythm["large_gaps"]]
        flags["RA.流利度"].append(f"🔴 意群内部有停顿：{'; '.join(gap_descs[:3])}")

    if rhythm.get("wpm_verdict") == "偏慢":
        flags["RA.流利度"].append(f"🟡 语速偏慢 ({rhythm['wpm']} WPM)")

    if rhythm["dur_cv"] > 0.6:
        flags["RA.流利度"].append("🟡 忽快忽慢：词长波动大")

    if rhythm["wpm"] > 220:
        flags["RA.流利度"].append("⚠️ 语速偏快，可能影响清晰度")

    # ── 发音 ──
    if rhythm["very_low_confidence_words"]:
        for w in rhythm["very_low_confidence_words"]:
            flags["RA.发音"].append(f"🔴 严重不清晰/吞音：'{w['word']}' (置信度 {w['prob']:.2f})")

    if rhythm["low_confidence_words"]:
        for w in rhythm["low_confidence_words"]:
            if w not in rhythm.get("very_low_confidence_words", []):
                flags["RA.发音"].append(f"🟡 元音不饱满/辅音不准：'{w['word']}' (置信度 {w['prob']:.2f})")

    # 从对比中提取更多发音问题
    if comparison:
        for c in comparison["word_comparisons"]:
            if c["verdict"] in ("🔴 明显拖长",) and c["word"].lower().strip(".,;:!?") not in FUNCTION_WORDS:
                flags["RA.发音"].append(f"🟡 拖音：'{c['word']}' (学生{c['student_dur']}s vs 母语{c['native_dur']}s)")
            if c.get("student_prob") and c["student_prob"] < 0.4:
                flags["RA.发音"].append(f"⚠️ 发音不准：'{c['word']}' (置信度 {c['student_prob']:.2f})")

    # ── 识别-原文偏差（有原文对齐时）──
    alignment_stats = whisper_result.get("alignment_stats")
    if alignment_stats:
        total = alignment_stats["total_original"]
        bad = alignment_stats["substitution"] + alignment_stats["missing"]
        if total > 0 and bad / total > 0.3:
            flags["RA.发音"].append(f"🔴 识别-原文偏差严重：{bad}/{total} 词不一致（替换 {alignment_stats['substitution']} + 缺失 {alignment_stats['missing']}）")
        elif bad > 0:
            flags["RA.发音"].append(f"🟡 识别-原文偏差：{bad} 词（替换 {alignment_stats['substitution']} + 缺失 {alignment_stats['missing']}）")
            # 列出具体的替换和缺失词
            words = whisper_result.get("words", [])
            subs = [w for w in words if w.get("match") == "substitution"]
            missing = [w for w in words if w.get("match") == "missing"]
            for w in subs[:3]:
                flags["RA.发音"].append(f"  替换: '{w['word']}' → Whisper识别为 '{w.get('whisper_word', '?')}'")
            for w in missing[:3]:
                flags["RA.发音"].append(f"  缺失: '{w['word']}' 未检测到")

    # ── CMU 音素级分析 ──
    if cmu_analysis:
        # 发音偏差定位（Whisper 替换 + 音素对比）
        phoneme_devs = cmu_analysis.get("phoneme_deviations", [])
        if phoneme_devs:
            for pd_item in phoneme_devs[:5]:
                diff = pd_item.get("phoneme_diff")
                if diff and diff.get("is_minimal_pair"):
                    diffs = diff.get("diff_positions", [])
                    for pos, t_ph, r_ph in diffs[:2]:
                        t_clean = t_ph.rstrip("012")
                        r_clean = r_ph.rstrip("012") if r_ph not in ("（省略）", "（插入）") else r_ph
                        note = ""
                        if t_clean in CN_DIFFICULT_PHONEMES:
                            note = f"（{CN_DIFFICULT_PHONEMES[t_clean].split('—')[0].strip()}）"
                        flags["RA.发音"].append(
                            f"🔴 发音偏差：'{pd_item.get('word', '?')}' → "
                            f"语音模型识别为'{pd_item.get('recognized_as', '?')}' "
                            f"｜疑错音素 /{t_clean}/ → /{r_clean}/ {note}")

        # 难点音素汇总
        diff_words = cmu_analysis.get("difficult_phoneme_words", [])
        if diff_words:
            from collections import Counter as _Counter
            ph_counter = _Counter()
            for dw in diff_words:
                if not dw.get("is_func"):
                    for d in dw.get("difficult", []):
                        ph_key = d.split("→")[0].strip() if "→" in d else d.split(" ")[0]
                        ph_counter[ph_key] += 1
            top_ph = ph_counter.most_common(3)
            if top_ph:
                summary = "、".join([f"{ph}({cnt}次)" for ph, cnt in top_ph])
                flags["RA.发音"].append(f"🟡 疑错音素（语音模型置信度低）：{summary} → 可能在这些音上发不准")

    # ── 语调（PTE 偏好稳定/偏平，不是问题）──
    if "变化偏大" in pitch.get("variation", ""):
        flags["RA.策略"].append(f"⚠️ 语调变化偏大 (CV={pitch.get('variation_cv', '?')}%) → PTE不需要戏剧化起伏")
    if "不稳定" in pitch.get("energy_stability", ""):
        flags["RA.策略"].append(f"⚠️ 能量不稳定 (RMS CV={pitch.get('energy_cv', '?')}%) → 忽大忽小")

    if "上扬" in pitch.get("tail_trend", ""):
        flags["RA.策略"].append("🔴 异常上扬：陈述句结尾往上走")

    # 去重
    for k in flags:
        flags[k] = list(dict.fromkeys(flags[k]))

    return flags


# ══════════════════════════════════════════════════
# 6. 生成预填诊断报告
# ══════════════════════════════════════════════════

def print_console_report(rhythm, pitch, flags, whisper_result, comparison, audio_path):
    """控制台彩色报告"""
    B = "\033[1m"
    R = "\033[91m"
    Y = "\033[93m"
    G = "\033[92m"
    C = "\033[96m"
    N = "\033[0m"

    print(f"\n{C}{'='*70}{N}")
    print(f"{C}  PTE 口语诊断 — AI 预分析报告 v2{N}")
    print(f"{C}{'='*70}{N}")
    print(f"\n{B}📁 音频:{N} {audio_path}")
    print(f"{B}📝 识别:{N} \"{whisper_result['text']}\"")
    print(f"{B}⏱ 语速:{N} {rhythm['wpm']} WPM ({rhythm['wpm_verdict']}) | {rhythm['total_words']}词 / {rhythm['audio_duration']}s")

    # ── 节奏诊断 ──
    print(f"\n{C}── 节奏 ──{N}")
    print(f"  词间平均间隙: {rhythm['mean_gap_ms']}ms")
    print(f"  虚实词时长比: {rhythm['content_func_ratio']}:1  {rhythm['ratio_verdict']}")
    print(f"  蹦词指数: {rhythm['staccato_score']}/6  → {B}{rhythm['staccato_verdict']}{N}")
    if rhythm["staccato_reasons"]:
        for r in rhythm["staccato_reasons"]:
            print(f"    ↳ {r}")

    if rhythm.get("dragged_func"):
        print(f"\n  {R}虚词拖音:{N}")
        for d in rhythm["dragged_func"]:
            print(f"    '{d['word']}' {d['duration']}s (均值{d['mean_dur']}s, {d['over_ratio']}x)")
    if rhythm.get("dragged_content"):
        print(f"\n  {R}实词拖音:{N}")
        for d in rhythm["dragged_content"]:
            print(f"    '{d['word']}' {d['duration']}s (均值{d['mean_dur']}s, {d['over_ratio']}x)")

    # ── 流利度 ──
    print(f"\n{C}── 流利度 ──{N}")
    if rhythm["noticeable_gaps"]:
        print(f"  明显间隙 ({rhythm['noticeable_gaps_count']} 处):")
        for g in rhythm["noticeable_gaps"][:5]:
            print(f"    ↳ {g['between']}: {g['gap_ms']}ms")
    else:
        print(f"  ✓ 无明显词间间隙")
    print(f"  词长变异系数: {rhythm['dur_cv']} (0.3-0.55 为自然区间)")

    # ── 发音 ──
    print(f"\n{C}── 发音 ──{N}")
    if rhythm["low_confidence_words"]:
        for w in rhythm["low_confidence_words"]:
            marker = "🔴" if w["prob"] < 0.3 else "🟡"
            print(f"  {marker} '{w['word']}' 置信度={w['prob']:.2f}")
    else:
        print(f"  ✓ 所有词识别置信度正常")

    # ── 语调 ──
    print(f"\n{C}── 语调 ──{N}")
    print(f"  基频: {pitch['mean_f0']}Hz (范围 {pitch['min_f0']}–{pitch['max_f0']}Hz)")
    print(f"  音高变化: {pitch['variation']} (CV={pitch.get('variation_cv', '?')}%)")
    print(f"  句尾趋势: {pitch['tail_trend']} ({pitch.get('tail_verdict', '?')})")

    # ── 母语者对比 ──
    if comparison:
        print(f"\n{C}── 母语者对比 ──{N}")
        if comparison.get("alignment_stats"):
            stats = comparison["alignment_stats"]
            print(f"  对齐: {stats['matched']}词 ✓ | 学生缺{stats['student_missing']} | 母语缺{stats['native_missing']}")
        print(f"  总时长比 (学生/母语): {comparison['overall_dur_ratio']}x")
        deviants = [c for c in comparison["word_comparisons"] if c["verdict"] not in ("✓",)]
        if deviants:
            for c in deviants:
                s_dur = f"{c['student_dur']}s" if c['student_dur'] else "?"
                n_dur = f"{c['native_dur']}s" if c['native_dur'] else "?"
                diff_str = f"(差{c['diff_ms']}ms)" if c['diff_ms'] is not None else ""
                extra = ""
                if c.get("student_heard_as") and c["student_heard_as"] != c["word"]:
                    extra += f" [学生被听成'{c['student_heard_as']}']"
                if c.get("native_heard_as") and c["native_heard_as"] != c["word"]:
                    extra += f" [母语被听成'{c['native_heard_as']}']"
                print(f"  {c['verdict']} '{c['word']}': 学生{s_dur} vs 母语{n_dur} {diff_str}{extra}")
        else:
            print(f"  ✓ 所有词在正常范围内")

    # ── 维度勾选 ──
    print(f"\n{C}── 诊断模板预填 ──{N}")
    for dim, items in flags.items():
        if items:
            print(f"\n  {B}【{dim}】{N}")
            for item in items:
                print(f"    [✓] {item}")
        else:
            print(f"\n  {B}【{dim}】{N}  [ ] 暂未检测到明显问题")

    print(f"\n{C}{'='*70}{N}")
    print(f"⚠️  AI 预分析仅供参考。根因判断、画像归类、处方仍需人工确认。")
    print(f"{C}{'='*70}{N}\n")


def _word_audio_uri(audio_array, sr, start, end):
    """提取词级音频片段（含前后缓冲），返回 base64 WAV data URI"""
    try:
        # 前后各加 80ms padding 防止 Whisper 切分偏差
        pad_ms = int(sr * 0.08)
        s = max(0, int(start * sr) - pad_ms)
        e = min(len(audio_array), int(end * sr) + pad_ms)
        if e - s < sr * 0.05:
            return None
        clip = audio_array[s:e]
        buf = io.BytesIO()
        sf.write(buf, clip, sr, format="WAV")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:audio/wav;base64,{b64}"
    except Exception:
        return None


def _build_reading_text(words, connected_speech):
    """
    构建朗读指导文本 — 标注连读‿、失爆、弱读（虚词灰色）。
    仅显示原文中的词，过滤多读词（extra），保持朗读指导干净。
    """
    if not words:
        return ""

    # 过滤多读词，保留原文词
    display_words = [w for w in words if w.get("match") != "extra"]

    # 建立 old_index → new_index 映射
    old_to_new = {}
    new_idx = 0
    for old_idx, w in enumerate(words):
        if w.get("match") != "extra":
            old_to_new[old_idx] = new_idx
            new_idx += 1

    # 重新计算连读/失爆标记（基于新的 display index）
    link_map = {}
    if connected_speech:
        for p in connected_speech:
            old_i = p["index"]
            if old_i in old_to_new:
                if p["type"] == "连读":
                    link_map[old_to_new[old_i]] = "‿"

    elision_letters = {}
    if connected_speech:
        for p in connected_speech:
            if p["type"] == "失爆" and "last_letter" in p:
                old_i = p["index"]
                if old_i in old_to_new:
                    elision_letters[old_to_new[old_i]] = (p["last_letter"], p["first_letter"])

    first_blue = set()
    last_blue = set()
    if connected_speech:
        for p in connected_speech:
            if p["type"] == "失爆":
                old_i = p["index"]
                if old_i in old_to_new and (old_i + 1) in old_to_new:
                    last_blue.add(old_to_new[old_i])
                    first_blue.add(old_to_new[old_i + 1])

    result_parts = []
    for i, w in enumerate(display_words):
        word_str = w["word"]
        clean = word_str.lower().strip(".,;:!?\"'")
        is_func = clean in FUNCTION_WORDS

        colored = word_str
        if i in last_blue and len(word_str) > 0:
            colored = word_str[:-1] + f'<span style="color:#3498db;font-weight:bold">{word_str[-1]}</span>'
        elif i in first_blue and len(word_str) > 0:
            colored = f'<span style="color:#3498db;font-weight:bold">{word_str[0]}</span>' + word_str[1:]

        if is_func and i not in last_blue and i not in first_blue:
            result_parts.append(f'<span style="color:#b0b0b0;font-size:0.9em">{colored}</span>')
        else:
            result_parts.append(f'<b>{colored}</b>')

        if i in link_map:
            result_parts.append(f'<span style="color:#e67e22;font-weight:bold;font-size:1.1em"> {link_map[i]} </span>')
        elif i < len(display_words) - 1:
            result_parts.append(" ")

    return "".join(result_parts)


def _analyze_connected_speech(words):
    """
    检测连读和失爆（基于拼写规则的启发式分析）。
    返回列表，每项描述一个词间发音现象。
    """
    # 元音字母开头 = 大概率元音开头
    def starts_with_vowel_sound(w):
        return w and w[0].lower() in "aeiou"

    # 词尾是否可能为辅音（排除不发音的 e）
    def ends_with_consonant_sound(w):
        if not w:
            return False
        w = w.lower()
        # 以不发音 e 结尾 → 看前一个字母
        if len(w) > 2 and w.endswith("e") and w[-2] not in "aeiou":
            return w[-2] in "bcdfghjklmnpqrstvwxyz"
        return w[-1] in "bcdfghjklmnpqrstvwxyz"

    # 词尾是否为爆破音 (p,b,t,d,k,g)
    def ends_with_stop(w):
        if not w:
            return False
        w = w.lower().rstrip(".,;:!?\"'")
        if not w:
            return False
        # 不发音 e 结尾的特殊处理
        last = w[-1]
        if last == "e" and len(w) > 2 and w[-2] in "bcdfghjklmnpqrstvwxyz":
            last = w[-2]
        return last in "pbtdkg"

    # 词首是否为爆破音
    def starts_with_stop(w):
        return w and w[0].lower() in "pbtdkg"

    phenomena = []

    for i in range(len(words) - 1):
        w1_raw = words[i]["word"]
        w2_raw = words[i+1]["word"]
        w1 = w1_raw.strip(".,;:!?\"'")
        w2 = w2_raw.strip(".,;:!?\"'")
        if not w1 or not w2:
            continue

        # 如果原文有标点分隔 → 不连读不失爆
        has_punct_between = (w1_raw.rstrip() != w1_raw and w1_raw.rstrip()[-1] in ".,;:!?")

        c1 = w1.lower()
        c2 = w2.lower()

        # 连读：辅+元 或 元+元（标点后不连读）
        if not has_punct_between:
            if ends_with_consonant_sound(c1) and starts_with_vowel_sound(c2):
                phenomena.append({
                    "type": "连读",
                    "between": f"{w1}‿{w2}",
                    "detail": f"{w1}尾辅音 + {w2}首元音 → 自然连读",
                    "index": i,
                })
            elif not ends_with_consonant_sound(c1) and starts_with_vowel_sound(c2):
                phenomena.append({
                    "type": "连读",
                    "between": f"{w1}‿{w2}",
                    "detail": f"元音+元音连读，可加 /j/ 或 /w/ 过渡",
                    "index": i,
                })

        # 失爆：爆破音+辅音开头（标点后不标）
        # 前词或后词是虚词 → 不标。虚词本就弱读/弱化，失爆在这里是自然现象，不需要刻意练习
        is_func_w1 = clean_word(w1) in FUNCTION_WORDS
        is_func_w2 = clean_word(w2) in FUNCTION_WORDS
        if not has_punct_between and ends_with_stop(c1) and not starts_with_vowel_sound(c2) and not is_func_w1 and not is_func_w2:
            last_letter = c1.rstrip("e")[-1] if c1.endswith("e") and len(c1)>2 and c1[-2] in "pbtdkg" else c1[-1]
            first_letter = c2[0]
            phenomena.append({
                "type": "失爆",
                "between": f"{w1} → {w2}",
                "detail": f"<span style='color:#3498db'>{last_letter}</span> → <span style='color:#3498db'>{first_letter}</span> 不完全爆破",
                "index": i,
                "last_letter": last_letter,
                "first_letter": first_letter,
            })

    return phenomena


def _build_color_text(words, word_diag, rhythm):
    """
    构建颜色编码的文本（HTML spans）。
    - 缺失词 → 红色 + 删除线
    - 替换词 → 红色（严重发音偏差）
    - 多读词 → 不显示（过滤掉，避免影响朗读指导/意群划分）
    - 虚词 → 浅灰色
    - 问题词 → error 红色，warning 橙色
    """
    if not word_diag:
        clean_words = [w for w in words if w.get("match") != "extra"]
        return " ".join([w["word"] for w in clean_words])

    # 建立 word → diag 映射（用 index）
    diag_map = {d["index"]: d for d in word_diag}

    parts = []
    for i, w in enumerate(words):
        match_status = w.get("match", "exact")
        # 多读词：不显示在原文标注中
        if match_status == "extra":
            continue

        word_str = w["word"]
        clean = w.get("clean") or word_str.lower().strip(".,;:!?\"'")
        is_func = clean in FUNCTION_WORDS
        d = diag_map.get(i)

        # 对齐状态着色
        if match_status == "missing":
            # 原文有但没读到 → 红色删除线
            parts.append(f'<span style="color:#e74c3c;text-decoration:line-through">{word_str}</span>')
            continue
        elif match_status in ("substitution", "fuzzy"):
            # 读偏了 → 红色，带识别结果提示
            whisper_w = w.get("whisper_word", "")
            title = f' title="→{whisper_w}"' if whisper_w and whisper_w != word_str else ""
            parts.append(f'<span style="color:#e74c3c"{title}>{word_str}</span>')
            continue

        # 正常匹配的词 → 按严重程度着色
        if d and d["severity"] == "error":
            color = "#e74c3c"
        elif d and d["severity"] == "warning":
            color = "#e67e22"
        elif is_func:
            color = "#b0b0b0"
        else:
            color = None

        if color:
            parts.append(f'<span style="color:{color}">{word_str}</span>')
        else:
            parts.append(word_str)

    return " ".join(parts)


# ══════════════════════════════════════════════════
# 8. 诊断波形图（波形 + AI 诊断标注叠加）
# ══════════════════════════════════════════════════

_font_configured = False

def _setup_chinese_font():
    """配置 matplotlib 中文字体，确保中文标签正常显示"""
    global _font_configured
    if _font_configured:
        return
    # 按优先级尝试可用中文字体
    candidates = [
        'Arial Unicode MS', 'STHeiti', 'Heiti TC', 'PingFang HK',
        'Hiragino Sans GB', 'Lantinghei SC', 'Noto Sans CJK SC',
        'SimHei', 'WenQuanYi Micro Hei',
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            _font_configured = True
            return
    # fallback: 尝试 sans-serif
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    _font_configured = True

def generate_diagnostic_waveform(audio, sr, words, word_diag, rhythm, student_name="", comparison=None):
    """
    生成叠加诊断标注的波形图。
    - 波形 + 词边界竖线（颜色=严重程度）
    - 词标签（颜色编码）
    - 间隙/蹦词标注
    - 问题标签
    返回 base64 PNG 字符串。
    """
    # ── 设置中文字体 ──
    _setup_chinese_font()

    # 只显示原文词（过滤 extra）
    display_words = [w for w in words if w.get("match") != "extra"]
    if not display_words:
        return None

    # 建立 index → diag 映射
    diag_map = {}
    diag_indices = {}  # match index → diag
    if word_diag:
        diag_map = {d["index"]: d for d in word_diag}
        # 同时按 clean word 建立查找（处理 index 不匹配的情况）
        for d in word_diag:
            key = d.get("clean", d["word"].lower().strip(".,;:!?\"'"))
            if key not in diag_indices:
                diag_indices[key] = d

    # ── 颜色函数 ──
    def get_word_color(w, idx):
        match = w.get("match", "exact")
        if match == "missing":
            return "#e74c3c"
        if match in ("substitution", "fuzzy"):
            return "#e74c3c"
        d = diag_map.get(idx)
        if d and d["severity"] == "error":
            return "#e74c3c"
        if d and d["severity"] == "warning":
            return "#e67e22"
        clean = w.get("clean") or w["word"].lower().strip(".,;:!?\"'")
        if clean in FUNCTION_WORDS:
            return "#b0b0b0"
        return "#2c3e50"

    def get_word_label(w, idx):
        """生成词标签：问题词在下方标注具体问题"""
        match = w.get("match", "exact")
        labels = []
        if match == "missing":
            labels.append("缺失")
        elif match in ("substitution", "fuzzy"):
            whisper_w = w.get("whisper_word", "")
            labels.append(f"→{whisper_w}" if whisper_w else "发音偏差")
        d = diag_map.get(idx)
        if d:
            for issue in d.get("issues", []):
                if issue == "✓ 正常":
                    continue
                # 提取简短标签
                if "严重偏离" in issue:
                    labels.append("严重偏离")
                elif "音节松散" in issue:
                    labels.append("音节松散")
                elif "拖音" in issue or "拖长" in issue:
                    labels.append("拖音")
                elif "元音" in issue or "辅音" in issue:
                    labels.append("发音不准")
                elif "蹦词" in issue or "间隙" in issue:
                    labels.append("蹦词")
                elif "卡顿" in issue:
                    labels.append("卡顿")
                elif "多音节" in issue:
                    labels.append("不清晰")
                elif "偏模糊" in issue:
                    labels.append("模糊")
        return labels[:2]  # 最多两个标签

    # ── 间隙数据 ──
    gap_after = {}  # word_original_index → gap_ms
    all_gaps = rhythm.get("noticeable_gaps", []) + rhythm.get("legit_pauses", [])
    for g in all_gaps:
        parts = g["between"].split(" → ")
        if len(parts) == 2:
            for i, w in enumerate(words):
                if i < len(words) - 1 and w["word"] == parts[0] and words[i+1]["word"] == parts[1]:
                    gap_after[i] = {
                        "gap_ms": g["gap_ms"],
                        "is_legit": g.get("is_sense_group_boundary", False),
                    }
                    break

    # ── 时间轴数据 ──
    times = []
    word_boundaries = []
    for w in display_words:
        if w.get("start") is not None and w.get("end") is not None:
            times.append(w["start"])
            times.append(w["end"])
            word_boundaries.append((w["start"], w["end"]))

    if not times:
        return None

    t_min = min(times)
    t_max = max(times)
    total_dur = t_max - t_min

    # 采样波形用于显示（降采样到显示精度）
    display_sr = 8000  # 8kHz 足够波形显示
    if sr != display_sr:
        audio_ds = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=display_sr)
    else:
        audio_ds = audio
    time_axis = np.linspace(t_min, t_max, int(total_dur * display_sr))

    # ── 绘图 ──
    fig, (ax_wave, ax_labels) = plt.subplots(
        2, 1, figsize=(14, 5.5),
        gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.08},
        facecolor='white'
    )

    # ── 上层：波形 ──
    ax_wave.plot(time_axis, audio_ds[:len(time_axis)], color='#34495e', linewidth=0.6, alpha=0.85)
    ax_wave.set_xlim(t_min, t_max)
    amp_max = max(abs(audio_ds[:len(time_axis)].min()), abs(audio_ds[:len(time_axis)].max()))
    ax_wave.set_ylim(-amp_max * 1.3, amp_max * 1.3)
    ax_wave.set_ylabel('振幅', fontsize=9, color='#7f8c8d')
    ax_wave.tick_params(axis='y', colors='#bdc3c7', labelsize=7)
    ax_wave.tick_params(axis='x', colors='#bdc3c7', labelsize=7)
    ax_wave.grid(True, alpha=0.2, color='#bdc3c7')
    ax_wave.set_facecolor('#fafbfc')

    # 词边界竖线 + 词区间背景色
    for i, w in enumerate(display_words):
        if w.get("start") is None or w.get("end") is None:
            continue
        s, e = w["start"], w["end"]
        color = get_word_color(w, i)
        alpha = 0.12 if color == "#b0b0b0" else 0.18
        # 半透明背景块
        ax_wave.axvspan(s, e, alpha=alpha, color=color, linewidth=0, zorder=1)
        # 词起始边界竖线
        if i > 0:
            ax_wave.axvline(x=s, color=color, linewidth=0.8, alpha=0.5, linestyle='-', zorder=2)

    # ── 间隙标注（在波形层）──
    for i, w in enumerate(display_words):
        if i in gap_after:
            g = gap_after[i]
            gms = g["gap_ms"]
            if g["is_legit"]:
                continue  # 合法意群停顿不标
            if gms > 120:
                # 在间隙处画标记
                if w.get("end") and i+1 < len(display_words) and display_words[i+1].get("start"):
                    gx = (w["end"] + display_words[i+1]["start"]) / 2
                    color = "#e74c3c" if gms >= 500 else "#e67e22"
                    ax_wave.annotate(
                        f'{gms}ms', xy=(gx, amp_max * 0.85),
                        fontsize=6, color=color, ha='center',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor=color, alpha=0.85, linewidth=0.8)
                    )

    # ── 标题 ──
    title = f"{student_name + ' — ' if student_name else ''}诊断波形"
    ax_wave.set_title(title, fontsize=12, fontweight='bold', color='#2c3e50', pad=8)

    # ── 下层：词标签 + 问题标签 ──
    ax_labels.set_xlim(t_min, t_max)
    ax_labels.set_ylim(0, 3)
    ax_labels.axis('off')

    # 词标签（第一行 y≈1.8）
    for i, w in enumerate(display_words):
        if w.get("start") is None or w.get("end") is None:
            continue
        mid = (w["start"] + w["end"]) / 2
        word_str = w["word"]
        color = get_word_color(w, i)
        match = w.get("match", "exact")
        fontweight = 'bold' if color not in ("#b0b0b0", "#2c3e50") else 'normal'

        # 缺失词加删除线效果（用 strikethrough 不支持，改用颜色+括号）
        if match == "missing":
            word_str = f"({word_str})"

        ax_labels.text(
            mid, 1.8, word_str, fontsize=8, color=color,
            ha='center', va='bottom', fontweight=fontweight,
            rotation=30 if len(word_str) > 6 else 0,
        )

    # 问题标签（第二行 y≈1.0）
    for i, w in enumerate(display_words):
        if w.get("start") is None or w.get("end") is None:
            continue
        mid = (w["start"] + w["end"]) / 2
        labels = get_word_label(w, i)
        if labels:
            for li, label in enumerate(labels):
                y = 1.0 - li * 0.55
                color = get_word_color(w, i)
                ax_labels.text(
                    mid, y, label, fontsize=5.5, color=color,
                    ha='center', va='top', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor=color, alpha=0.7, linewidth=0.5)
                )

    # ── 底部图例 ──
    legend_y = -0.2
    legend_items = [
        ("🔴 严重问题", "#e74c3c"),
        ("🟡 警告", "#e67e22"),
        ("灰色 虚词弱读", "#b0b0b0"),
        ("黑色 实词正常", "#2c3e50"),
    ]
    legend_text = "  |  ".join([t for t, _ in legend_items])
    ax_labels.text(
        (t_min + t_max) / 2, legend_y, legend_text,
        fontsize=7, color='#95a5a6', ha='center', va='top'
    )

    # ── 统计信息 ──
    stats_text = (
        f"WPM: {rhythm.get('wpm', '?')}  |  "
        f"虚实比: {rhythm.get('content_func_ratio', '?')}:1  |  "
        f"蹦词指数: {rhythm.get('staccato_score', '?')}/6  |  "
        f"卡顿: {rhythm.get('hesitation_count', 0)}处"
    )
    ax_labels.text(
        (t_min + t_max) / 2, legend_y - 0.45, stats_text,
        fontsize=7, color='#7f8c8d', ha='center', va='top',
        style='italic'
    )

    plt.tight_layout(pad=1.5)

    # 保存为 base64 PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ══════════════════════════════════════════════════
# 9. 三线音高图（声调层级可视化）
# ══════════════════════════════════════════════════

def _build_pitch_tiers_html(words, word_diag=None, pitch=None, native_words=None):
    """
    生成两组三线音高图：
    ① 学生实际 — 全部用真实声学数据分层
    ② 母语者参考 — 用语言学规则（虚词弱读、实词重读）
    """
    display_words = [w for w in words if w.get("match") != "extra"]
    if not display_words:
        return ""

    # 实词时长均值
    content_durs = []
    for w in display_words:
        clean = w.get("clean") or w["word"].lower().strip(".,;:!?\"'")
        if clean not in FUNCTION_WORDS and w.get("duration"):
            content_durs.append(w["duration"])
    dur_mean = np.mean(content_durs) if content_durs else 0.3

    # 建立 word_diag 索引
    diag_map = {}
    if word_diag:
        for d in word_diag:
            diag_map[d["index"]] = d

    # ── ① 学生实际分层：全用真实数据 ──
    student_tiers = []
    for i, w in enumerate(display_words):
        clean = w.get("clean") or w["word"].lower().strip(".,;:!?\"'")
        dur = w.get("duration")
        match = w.get("match", "exact")
        d = diag_map.get(i, {})

        if match == "missing":
            student_tiers.append((w["word"], 1))
            continue

        # 用真实数据判断：音高偏高 → 强调；音高偏低/虚词低 → 弱读；其他 → 标准
        issues = d.get("issues", [])
        has_high_pitch = any("音高偏高" in iss for iss in issues)
        has_low_pitch = any("音高偏低" in iss for iss in issues)
        is_func = clean in FUNCTION_WORDS

        if has_high_pitch:
            student_tiers.append((w["word"], 3))  # 真实音高偏高 → 强调
        elif has_low_pitch and not is_func:
            student_tiers.append((w["word"], 1))  # 实词音高偏低 → 弱读
        elif dur and dur > dur_mean * 1.3:
            student_tiers.append((w["word"], 3))  # 明显拉长 → 强调
        elif is_func:
            student_tiers.append((w["word"], 1))  # 虚词默认弱读
        else:
            student_tiers.append((w["word"], 2))  # 普通实词 → 标准

    # ── ② 母语者参考分层：语言学理想规则 ──
    native_tiers = []
    for i, w in enumerate(display_words):
        clean = w.get("clean") or w["word"].lower().strip(".,;:!?\"'")
        is_func = clean in FUNCTION_WORDS

        if is_func:
            native_tiers.append((w["word"], 1))  # 虚词 → 弱读
        elif len(clean) >= 6:
            native_tiers.append((w["word"], 3))  # 多音节实词 → 强调
        else:
            native_tiers.append((w["word"], 2))  # 其余实词 → 标准

    # 构建 HTML
    # 计算时间轴（总时长用于百分比定位）
    valid_times = [w for w in display_words if w.get("start") is not None and w.get("end") is not None]
    if not valid_times:
        return ""
    t_min = min(w["start"] for w in valid_times)
    t_max = max(w["end"] for w in valid_times)
    total_dur = t_max - t_min
    if total_dur <= 0:
        return ""

    tier_config = {
        3: ("强调", "#e67e22", "2px solid #e67e22"),
        2: ("标准", "#ccc", "1px solid #555"),
        1: ("弱读", "#888", "1px dashed #444"),
    }

    px_per_sec = 250
    container_px = max(600, int(total_dur * px_per_sec))

    def _render_tier_set(label_title, tiers_data, label_color, time_words):
        """渲染一组三线音高图。time_words 提供时间位置。"""
        # 计算本组的时间范围
        tw_valid = [tw for tw in time_words if tw.get("start") is not None]
        if not tw_valid:
            return ""
        tw_min = min(tw["start"] for tw in tw_valid)
        tw_max = max(tw["end"] for tw in tw_valid if tw.get("end") is not None)
        tw_dur = tw_max - tw_min
        if tw_dur <= 0:
            tw_dur = total_dur
        tw_container_px = max(600, int(tw_dur * px_per_sec))

        rows = [
            f'<div style="color:{label_color};font-size:0.72em;font-weight:bold;'
            f'margin-bottom:4px;padding-left:2px">{label_title} &nbsp;|&nbsp;'
            f'强调=重读拉长 &nbsp;|&nbsp; 标准=正常实词 &nbsp;|&nbsp; 弱读=虚词轻短 &nbsp;|&nbsp;'
            f'{tw_dur:.1f}s</div>',
        ]
        for tier_num in [3, 2, 1]:
            name, color, border = tier_config[tier_num]
            rows.append(
                f'<div style="position:relative;height:26px;margin:1px 0;'
                f'border-bottom:{border};width:{tw_container_px}px;min-width:100%">'
                f'<span style="position:absolute;left:2px;top:50%;'
                f'transform:translateY(-50%);color:{color};'
                f'font-size:0.62em;font-weight:bold">{name}</span>'
            )
            for idx, tw in enumerate(time_words):
                if tw.get("start") is None:
                    continue
                if tiers_data[idx][1] != tier_num:
                    continue
                left_px = int((tw["start"] - tw_min) * px_per_sec) + 40
                rows.append(
                    f'<span style="position:absolute;left:{left_px}px;'
                    f'top:50%;transform:translateY(-50%);'
                    f'color:{color};font-weight:bold;font-size:0.76em;'
                    f'white-space:nowrap">{tw["word"]}</span>'
                )
            rows.append('</div>')
        # 时间刻度
        n_ticks = min(6, max(2, int(tw_dur)))
        ticks = []
        for i in range(n_ticks + 1):
            t = tw_min + tw_dur * i / n_ticks
            left_px = int((t - tw_min) * px_per_sec) + 40
            ticks.append(
                f'<span style="position:absolute;left:{left_px}px;top:0;'
                f'color:#555;font-size:0.5em">{t:.1f}s</span>'
            )
        rows.append(
            f'<div style="position:relative;height:12px;width:{tw_container_px}px;'
            f'min-width:100%;border-top:1px solid #333">{"".join(ticks)}</div>'
        )
        return "\n".join(rows)

    # 母语者词列表（有参考录音用真实时间，无则按经验值估算）
    if native_words:
        native_time_words = native_words
    else:
        # 按语言学经验值估算母语者理想节奏
        native_time_words = []
        t = 0.0
        for w in display_words:
            clean = w.get("clean") or w["word"].lower().strip(".,;:!?\"'")
            is_func = clean in FUNCTION_WORDS
            n_chars = len(clean)
            if is_func:
                dur = 0.10 + n_chars * 0.015  # 虚词极短：~0.12–0.25s
            elif n_chars >= 7:
                dur = 0.28 + n_chars * 0.025  # 长实词：~0.45–0.65s
            elif n_chars >= 5:
                dur = 0.22 + n_chars * 0.025  # 中实词：~0.35–0.50s
            else:
                dur = 0.18 + n_chars * 0.025  # 短实词：~0.20–0.35s
            gap = 0.025  # 母语者词间间隙极短 ~25ms
            native_time_words.append({
                "word": w["word"],
                "start": t,
                "end": t + dur,
            })
            t += dur + gap
        # 末尾不加间隙
        if native_time_words:
            native_time_words[-1]["end"] = native_time_words[-1]["start"] + (
                0.3 if len(native_time_words[-1]["word"]) > 5 else 0.2
            )

    # ── 原文参照行（所有词按学生时间排列）──
    ref_row_html = ""
    if display_words:
        ref_parts = []
        ref_parts.append(
            f'<div style="position:relative;height:24px;margin:1px 0;'
            f'width:{container_px}px;min-width:100%;border-bottom:1px solid #555">'
            f'<span style="position:absolute;left:2px;top:50%;transform:translateY(-50%);'
            f'color:#777;font-size:0.6em;font-weight:bold">原文</span>'
        )
        for w in display_words:
            if w.get("start") is None:
                continue
            left_px = int((w["start"] - t_min) * px_per_sec) + 40
            ref_parts.append(
                f'<span style="position:absolute;left:{left_px}px;'
                f'top:50%;transform:translateY(-50%);'
                f'color:#aaa;font-size:0.76em;white-space:nowrap">{w["word"]}</span>'
            )
        ref_parts.append('</div>')
        ref_row_html = "\n".join(ref_parts)

    html = [
        '<div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;',
        'margin:10px 0;padding:10px 14px;background:#1a1a1a;',
        'border-radius:6px;overflow:auto;border-left:3px solid #e67e22">',
        '<div style="color:#e67e22;font-size:0.75em;margin-bottom:8px">',
        '📊 声调层级（三线音高图）</div>',
        ref_row_html,
        _render_tier_set("🎙️ 学生实际", student_tiers, "#e67e22", display_words),
        '<div style="margin:12px 0;border-top:1px solid #444"></div>',
        _render_tier_set("🎯 母语者参考", native_tiers, "#3498db", native_time_words),
        '</div>',
    ]
    return "\n".join(html)


def _build_cmu_section(cmu_analysis):
    """构建 CMU 音素级分析报告段落（教师专属区域）。"""
    if not cmu_analysis:
        return ""

    stress_errors = cmu_analysis.get("stress_errors", [])
    phoneme_devs = cmu_analysis.get("phoneme_deviations", [])
    diff_words = cmu_analysis.get("difficult_phoneme_words", [])

    if not stress_errors and not phoneme_devs and not diff_words:
        return ""

    lines = []
    lines.append("## 🧬 音素级分析（标准发音词典）")
    lines.append("")
    lines.append("> 基于标准发音词典（134k 词条）的音素序列，定位重音和发音偏差。")
    lines.append("")

    # ── 重音位置错误 ──
    if stress_errors:
        lines.append("### 💡 重音参考（标准发音词典）")
        lines.append("")
        lines.append("| 词 | 音节数 | 主重音 | 音素序列 |")
        lines.append("|-----|--------|--------|---------|")
        for se in stress_errors[:10]:
            lines.append(
                f"| **{se.get('word', '?')}** | {se.get('n_syllables', '?')} | "
                f"第{se.get('expected_primary_syl', '?')}音节 | "
                f"`{se.get('phoneme_sequence', '')}` |"
            )
        lines.append("")

    # ── 发音偏差定位 ──
    if phoneme_devs:
        lines.append("### 🔍 发音偏差定位（语音模型替换 + 音素对比）")
        lines.append("")
        lines.append("> 语音模型把目标词识别成了另一个词 → 对比两个词的音素序列，推断具体哪个音出错了。")
        lines.append("")
        lines.append("| 目标词 | 识别为 | 疑错音素 | 目标 → 实际 | 说明 |")
        lines.append("|--------|--------|---------|------------|------|")
        for pd_item in phoneme_devs[:10]:
            diff = pd_item.get("phoneme_diff")
            if not diff:
                continue
            diffs = diff.get("diff_positions", [])
            diff_strs = []
            diff_notes = []
            for pos, t_ph, r_ph in diffs[:2]:
                t_clean = t_ph.rstrip("012")
                r_clean = r_ph.rstrip("012") if r_ph not in ("（省略）", "（插入）") else r_ph
                diff_strs.append(f"/{t_clean}/ → /{r_clean}/")
                if t_clean in CN_DIFFICULT_PHONEMES:
                    diff_notes.append(CN_DIFFICULT_PHONEMES[t_clean].split("—")[0].strip())
            lines.append(
                f"| **{pd_item.get('word', '?')}** | "
                f"{pd_item.get('recognized_as', '?')} | "
                f"{'<br>'.join(d for d in diff.get('target_difficult', [])[:2]) or '—'} | "
                f"{'<br>'.join(diff_strs)} | "
                f"{'; '.join(diff_notes) if diff_notes else '—'} |"
            )
        lines.append("")

    # ── 疑错音素（仅低置信度词，有实际证据）──
    if diff_words:
        content_diff = [dw for dw in diff_words if not dw.get("is_func")][:10]
        if content_diff:
            lines.append("### 🟡 疑错音素（语音模型置信度低 + 含难点音素）")
            lines.append("")
            lines.append("> 以下词语音模型识别置信度 < 0.5，且包含中国学习者高频错音。")
            lines.append("> **不是扫射所有难点音素——只标了有实际发音证据的词。**")
            lines.append("")
            lines.append("| 词 | 置信度 | 音素序列 | 疑错音素 |")
            lines.append("|-----|--------|---------|---------|")
            for dw in content_diff:
                lines.append(
                    f"| **{dw.get('word', '?')}** | "
                    f"{dw.get('prob', '?')} | "
                    f"`{dw.get('phonemes', '')}` | "
                    f"{'<br>'.join(dw.get('difficult', [])[:3])} |"
                )
            lines.append("")

    return "\n".join(lines)


def generate_markdown_draft(audio_path, rhythm, pitch, flags, whisper_result, comparison, student_name="", word_diag=None, connected_speech=None, waveform_b64=None, native_words=None, cmu_analysis=None, template_analysis=None, question_type="RA"):
    """生成诊断报告 — 只显示检测到的问题，不列出未触发的项"""
    original_text = whisper_result.get("original_text")
    alignment_stats = whisper_result.get("alignment_stats")
    words = whisper_result["words"]

    lines = []
    lines.append(f"# {student_name or '学生'} — PTE 口语诊断")
    lines.append("")
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"**{now}**")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 对齐统计（有原文时）──
    if alignment_stats:
        s = alignment_stats
        total = s["total_original"]
        match_pct = round(s["matched"] / total * 100) if total > 0 else 0
        lines.append("## 📋 识别 vs 原文")
        lines.append("")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 匹配率 | {match_pct}% ({s['matched']}/{total} 词) |")
        if s["substitution"]:
            lines.append(f"| 🔴 替换词 | {s['substitution']} 词（发音偏差大，语音模型识别为其他词）|")
        if s["missing"]:
            lines.append(f"| 🔴 缺失词 | {s['missing']} 词（原文有但录音中未检测到）|")
        if s["extra"]:
            lines.append(f"| ⚠️ 多读词 | {s['extra']} 词（录音中有但原文没有）|")
        lines.append("")
        lines.append("---")
        lines.append("")

    # ── 诊断波形图 ──
    if waveform_b64:
        lines.append("## 📈 诊断波形")
        lines.append("")
        lines.append(f'<img src="data:image/png;base64,{waveform_b64}" '
                     f'style="width:100%;border:1px solid #e0e0e0;border-radius:6px" '
                     f'alt="诊断波形图" />')
        lines.append("")
        lines.append('<details>')
        lines.append('<summary><b>📖 如何看懂这张波形图？（点击展开）</b></summary>')
        lines.append("")
        lines.append("### 🔍 三个视觉直觉")
        lines.append("")
        lines.append("**1. 高低 = 轻重**")
        lines.append("- 波峰越高 → 声音越响 → 这个词读得越重")
        lines.append("- 灰色词（the, a, of, in...）应该趴在地上，几乎看不见波峰")
        lines.append("- 深色实词应该有明显的起伏")
        lines.append("- 英语节奏靠「实词重、虚词轻」，看波形就能判断有没有做到")
        lines.append("")
        lines.append("**2. 间距 = 连贯度**")
        lines.append("- 词和词之间如果有红/橙色数字（如 `350ms`），说明这里断了")
        lines.append("- 正常说话词之间是连着的，不该停的地方停了 = 蹦词")
        lines.append("- 就像中文读成「我/是/学生」一字一顿——英语母语者听感是机器人在说话")
        lines.append("")
        lines.append("**3. 颜色 = 问题在哪**")
        lines.append("")
        lines.append("| 颜色 | 含义 | 怎么看 |")
        lines.append("|------|------|--------|")
        lines.append("| 🔴 红色 | 严重问题 | 发音偏差大 / 吞音 / 卡顿 |")
        lines.append("| 🟠 橙色 | 警告 | 拖音 / 模糊 / 不该停的小停顿 |")
        lines.append("| ⬛ 深色 | 正常实词 | 读得不错 |")
        lines.append("| ⬜ 灰色 | 虚词弱读 | 虚词就该轻，灰色是**好事** |")
        lines.append("")
        lines.append("### 🎯 高分波形的「稳定」是什么意思？")
        lines.append("")
        lines.append("PTE 高分的波形看起来稳定，**但稳定 ≠ 平坦**：")
        lines.append("")
        lines.append("| 高分特征 | 波形表现 | 你的自查 |")
        lines.append("|---------|---------|---------|")
        lines.append("| 能量一致 | 从头到尾波峰高度差不多 | 不会第一句大声第二句虚了 |")
        lines.append("| 节奏规律 | 实词高→虚词低→实词高→虚词低，交替有规律 | 不是忽大忽小，是有预期的节奏 |")
        lines.append("| 连贯不断 | 词之间平滑连接，没有断崖式缝隙 | 像一条连续的河，不像一串孤立的珠子 |")
        lines.append("")
        lines.append("> 💡 **一句话记住**：高分波形像一条平滑的河——有自然的波浪，但不会突然瀑布、突然干涸。**有起伏但不失控，有轻重但不断裂。**")
        lines.append("")
        lines.append("</details>")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## 📋 基础数据")
    lines.append("")
    lines.append(f"| 指标 | 数值 | 判断 |")
    lines.append(f"|------|------|------|")

    # ── 标点恢复（在构建颜色文本之前，确保"识别文本"行带标点）──
    #    Whisper 对长音频（SGD）常不输出标点，提前恢复并同步到 words 数组。
    import re as re_mod
    _punct_restored = False
    if not original_text:
        _raw_text = whisper_result.get("text", "").strip()
        if _raw_text:
            _raw_text = _raw_text.replace("'", "'").replace('"', '"')
            _raw_text = _raw_text.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
            _restored_text, _punct_restored = _restore_punctuation(_raw_text)
            if _punct_restored:
                whisper_result["text"] = _restored_text
                _sync_punctuation_to_words(whisper_result["words"], _restored_text)

    # 颜色编码文本（基于原文/对齐后的词）
    color_text = _build_color_text(words, word_diag, rhythm)
    row_label = "原文（诊断标注）" if original_text else "识别文本"
    lines.append(f"| {row_label} | {color_text} | — |")
    # 如果有原文且识别不同，额外显示识别文本（只显示原文对应的词，过滤多读词）
    if original_text and alignment_stats and (alignment_stats["substitution"] + alignment_stats["missing"] > 0):
        whisper_raw = " ".join([w.get("whisper_word") or w["word"] for w in words if w.get("match") != "extra"])
        lines.append(f"| AI 识别 | {whisper_raw} | 与原文不一致的词已标红 |")
    # 朗读指导文本（总是显示，即使没有连读失爆也标注弱读）
    reading_text = _build_reading_text(words, connected_speech)
    guide_note = "<b>粗体</b>=实词 <span style='color:#b0b0b0'>灰色</span>=弱读"
    if connected_speech:
        guide_note += " <span style='color:#e67e22'>‿</span>=连读 <span style='color:#3498db;font-weight:bold'>蓝</span>=失爆"
    # AI 示范朗读（拆句 → 逐句 Edge TTS 生成 → 内嵌播放器，国内可访问 + 自然发音）
    demo_audio_html = ""
    try:
        tts_full = (original_text or whisper_result.get("text", "")).strip()  # TTS 用原文或已恢复标点的识别文本
        # 清理 Unicode 引号
        tts_full = tts_full.replace("'", "'").replace('"', '"')
        tts_full = tts_full.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
        # 标点恢复（可能已在上面做过，第二次调用会因已有标点而跳过）
        tts_full, _ = _restore_punctuation(tts_full)
        final_sents = _split_text_for_tts(tts_full, max_chars=180)

        if final_sents:
            audio_tags = []
            for i, sent in enumerate(final_sents):
                tag = None
                mp3_bytes = _edge_tts(sent)
                if mp3_bytes and len(mp3_bytes) > 500:
                    b64 = base64.b64encode(mp3_bytes).decode()
                    tag = (
                        f' <audio controls style="height:1.4em;width:130px;display:inline-block;vertical-align:middle" '
                        f'src="data:audio/mpeg;base64,{b64}" preload="none" title="句{i+1}"></audio>'
                    )
                if tag:
                    audio_tags.append(f'<span style="white-space:nowrap">▶{i+1}{tag}</span>')
                else:
                    audio_tags.append(f'<span style="color:#999">▶{i+1}(无音频)</span>')
            has_audio = any('audio controls' in t for t in audio_tags)
            if has_audio:
                demo_audio_html = (
                    f'<br><br>🎧 示范朗读：'
                    f'<span style="font-size:0.9em">{" ".join(audio_tags)}</span>'
                )
            else:
                demo_audio_html = f'<br><br>⚠️ 示范朗读暂不可用（TTS 服务异常）'
    except Exception:
        demo_audio_html = f'<br><br>⚠️ 示范朗读暂不可用'


    lines.append(f"| 朗读指导 | {reading_text}{demo_audio_html} | {guide_note} |")
    lines.append(f"| 语速 | {rhythm['wpm']} WPM | {rhythm['wpm_verdict']} |")
    lines.append("")

    # ── 意群分析：独立段落（避免长文本撑破表格）──
    actual_text = rhythm.get("actual_pauses_text", "") or rhythm.get("annotated_text", "")
    suggested_text = rhythm.get("suggested_groups_text", "")
    mismatches = rhythm.get("sense_group_mismatches", [])

    if actual_text or suggested_text:
        lines.append("## 🗣️ 意群分析")
        lines.append("")

    if actual_text:
        lines.append(f"**实际停顿**（你的真实停顿位置）：")
        lines.append(f"> {actual_text}")
        lines.append('> `/` 短停 (120–300ms)　`//` 长停 (>300ms)　→ 对照下方建议意群判断是否合理')
        lines.append("")

    if suggested_text:
        lines.append(f"**建议意群**（基于语法结构的自然切分）：")
        lines.append(f"> {suggested_text}")
        lines.append('> <span style="color:#1565C0"><b>/</b></span> = 建议的意群边界')
        lines.append("")
    elif actual_text:
        lines.append(f"**建议意群**：")
        lines.append("> 💡 上传**原文**可获取基于语法结构的意群建议（当前未提供原文，无法分析）")
        lines.append("")

    if mismatches:
        lines.append(f"**意群对比**（实际 vs 建议）：")
        for m in mismatches[:5]:
            lines.append(f"- {m['desc']}：{m['at']} ({m['gap_ms']}ms)")
        if len(mismatches) > 5:
            lines.append(f"- ...共 {len(mismatches)} 处不匹配")
        lines.append("")
    # 以下指标详见 Web 界面「节奏详情」Tab
    lines.append("")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 教师专属区域（不出现在学生导出 PNG 中）──
    lines.append('<div class="teacher-only">')
    # ── 三线音高图 ──
    pitch_tiers_html = _build_pitch_tiers_html(words, word_diag, pitch, native_words)
    if pitch_tiers_html:
        lines.append("## 📊 声调层级（三线音高图）")
        lines.append("")
        lines.append("> 每个词放在它所属的声调线上。弱读线=虚词轻短，标准线=正常实词，强调线=重读拉长。")
        lines.append("> 学生可以看着这张图练节奏——眼睛跟着三条线走，嘴巴就知道哪里轻哪里重。")
        lines.append("")
        lines.append(pitch_tiers_html)
        lines.append("")
        lines.append("---")
        lines.append("")

    # ── CMU 音素级分析 ──
    cmu_section = _build_cmu_section(cmu_analysis)
    if cmu_section:
        lines.append(cmu_section)
        lines.append("")
        lines.append("---")
        lines.append("")

    # ── 逐词诊断（核心输出）──
    if word_diag:
        lines.append("## 🔍 逐词诊断")
        lines.append("")
        lines.append("> 每个词的具体问题。只列出有问题的词，正常的省略。")
        lines.append("")

        problem_words = [w for w in word_diag if w['severity'] != 'ok']
        if problem_words:
            lines.append("| # | 词 | 时长 | 置信度 | 参考 | 类型 | 具体问题 |")
            lines.append("|---|-----|------|--------|------|------|---------|")
            for w in problem_words:
                ref_str = f"{w['ref_duration']}s" if w['ref_duration'] else "—"
                word_type = "虚词" if w['is_function_word'] else "实词"
                dur_str = f"{w['duration']}s" if w.get('duration') is not None else "—"
                prob_str = f"{w['prob']:.0%}" if w.get('prob') is not None else "—"
                issue_list = [i for i in w['issues'] if i != "✓ 正常"]
                issue_text = "<br>".join(issue_list)
                # 实词问题加发音链接（字典 + 学生）
                word_display = f"**{w['word']}**"
                if not w.get('is_function_word', False):
                    if w.get("dict_audio_uri"):
                        word_display += f' 📖<audio controls style="height:1.2em;width:80px;display:inline-block;vertical-align:middle" src="{w["dict_audio_uri"]}" title="字典发音" preload="none"></audio>'
                    if w.get("student_audio_uri"):
                        word_display += f' 🎤<audio controls style="height:1.2em;width:80px;display:inline-block;vertical-align:middle" src="{w["student_audio_uri"]}" title="学生发音"></audio>'
                lines.append(
                    f"| {w['index']+1} | {word_display} | {dur_str} "
                    f"| {prob_str} | {ref_str} | {word_type} | {issue_text} |"
                )
            lines.append("")
        else:
            lines.append("✓ 所有词在正常范围内")
            lines.append("")

    # ── 连读/失爆 ──
    if connected_speech:
        linking = [p for p in connected_speech if p["type"] == "连读"]
        elision = [p for p in connected_speech if p["type"] == "失爆"]
        if linking or elision:
            lines.append("---")
            lines.append("")
            lines.append("## 🔗 连读 & 失爆")
            lines.append("")
            lines.append("> 标注文本中应发生连读和失爆的位置，帮助学生练习自然语流。")
            lines.append("")

            if linking:
                lines.append("### 连读")
                lines.append("| 位置 | 说明 |")
                lines.append("|------|------|")
                for p in linking:
                    lines.append(f"| {p['between']} | {p['detail']} |")
                lines.append("")

            if elision:
                lines.append("### 失爆（不完全爆破）")
                lines.append("| 位置 | 说明 |")
                lines.append("|------|------|")
                for p in elision:
                    lines.append(f"| {p['between']} | {p['detail']} |")
                lines.append("")

    lines.append('</div>')  # end teacher-only
    # ── 整体问题汇总（学生也可见）──
    lines.append("---")
    lines.append("")
    lines.append("## 📊 整体诊断")
    lines.append("")

    all_dims = [
        ("节奏", "RA.节奏"),
        ("流利度", "RA.流利度"),
        ("发音", "RA.发音"),
        ("语调/策略", "RA.策略"),
    ]

    has_any_flag = False
    for dim_name, dim_key in all_dims:
        items = flags.get(dim_key, [])
        if items:
            has_any_flag = True
            lines.append(f"### {dim_name}")
            for item in items:
                lines.append(f"- {item}")
            lines.append("")

    if not has_any_flag:
        lines.append("未检测到明显的整体性问题。请参考逐词诊断。")
        lines.append("")

    # ── 节奏详细证据 ──
    if rhythm["staccato_reasons"]:
        lines.append("### 节奏证据")
        for r in rhythm["staccato_reasons"]:
            lines.append(f"- {r}")
        lines.append("")

    if rhythm.get("dragged_func"):
        lines.append("### 虚词拖音")
        for d in rhythm["dragged_func"]:
            lines.append(f"- **{d['word']}** — {d['duration']}s（均值 {d['mean_dur']}s，{d['over_ratio']}x → 拖虚词想实词）")
        lines.append("")

    if rhythm.get("dragged_content"):
        lines.append("### 实词拖音")
        for d in rhythm["dragged_content"]:
            lines.append(f"- **{d['word']}** — {d['duration']}s（均值 {d['mean_dur']}s，{d['over_ratio']}x → 元音/重读音节拉得过长）")
        lines.append("")

    # ── 母语者对比 ──
    if comparison:
        lines.append("---")
        lines.append("")
        lines.append("## ⚖️ 母语者对比")
        lines.append("")
        total_ratio = comparison.get('overall_dur_ratio', 1)
        faster_slower = "偏慢" if total_ratio > 1.1 else "偏快" if total_ratio < 0.9 else "接近"
        lines.append(f"学生总时长 / 参考总时长 = **{total_ratio}x**（{faster_slower}）")
        lines.append("")

        if comparison.get("alignment_stats"):
            stats = comparison["alignment_stats"]
            lines.append(f"对齐: {stats['matched']} 词匹配 | 学生缺 {stats['student_missing']} | 参考缺 {stats['native_missing']}")
            lines.append("")

        deviant_words = [c for c in comparison["word_comparisons"] if c["verdict"] not in ("✓",)]
        if deviant_words:
            lines.append("| 词 | 学生 | 参考 | 差异 | 判断 |")
            lines.append("|----|------|------|------|------|")
            for c in deviant_words:
                s_dur = f"{c['student_dur']}s" if c['student_dur'] else "—"
                n_dur = f"{c['native_dur']}s" if c['native_dur'] else "—"
                diff = f"{c['diff_ms']}ms" if c.get('diff_ms') is not None else "—"
                lines.append(f"| {c['word']} | {s_dur} | {n_dur} | {diff} | {c['verdict']} |")
            lines.append("")

    # ── 模板/内容比例（DI/RL/SGD 专项）──
    if template_analysis:
        ta = template_analysis
        lines.append("---")
        lines.append("")
        lines.append("## 📝 模板/内容分析")
        lines.append("")
        t_pct = round(ta["template_ratio"] * 100)
        d_pct = round(ta["template_duration_ratio"] * 100)
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 判定 | {ta['icon']} **{ta['verdict']}** |")
        lines.append(f"| 模板词占比 | {t_pct}%（{ta['template_word_count']}/{ta['total_words']} 词）|")
        lines.append(f"| 内容词占比 | {100-t_pct}%（{ta['content_word_count']}/{ta['total_words']} 词）|")
        lines.append(f"| 模板时长占比 | {d_pct}% |")
        lines.append("")

        if ta["verdict"] in ("偏模板", "模板过多"):
            lines.append("> ⚠️ 模板词占比偏高。PTE 评分中，过多的套话会压低内容分和流利度分。")
            lines.append("> 建议：缩短模板句型、增加对具体数据/内容的描述比例。")
            lines.append("")

        # 列出检测到的模板片段
        if ta.get("segments"):
            unique = []
            seen = set()
            for _, phrase in ta["segments"]:
                if phrase not in seen:
                    unique.append(phrase)
                    seen.add(phrase)
            if unique:
                lines.append("**检测到的模板片段**:")
                lines.append("")
                for phrase in unique[:10]:
                    lines.append(f"- `{phrase}`")
                if len(unique) > 10:
                    lines.append(f"- ... 共 {len(unique)} 段")
                lines.append("")

    lines.append("*AI 辅助生成*")

    return "\n".join(lines)


# ══════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════

def run_analysis(audio_path, ref_path=None, text=None, model_name="base", student_name="", question_type="RA"):
    """
    核心分析管道（供 CLI 和 Web 界面复用）。
    返回 dict: {markdown, md_path, json_path, whisper_result, rhythm, pitch, comparison, flags, question_type}
    不做任何 print。
    """
    # 1. 加载学生音频
    audio, sr = load_audio(audio_path)

    # 1.5 检测真实语音起点（不改音频，仅用于后续修正首词 duration）
    #     场景：学员开口前有长停顿 → Whisper 可能把部分静音计入首词的 start，
    #     使 duration = end - start 虚高 → 首词（尤其虚词如 "in"）被误判为"拖音"。
    #     用能量检测找到真实语音起点，传给 rhythm/diagnose 做修正。
    _, trim_indices = librosa.effects.trim(audio, top_db=25)
    speech_onset = trim_indices[0] / sr if sr > 0 else 0.0
    # 若几乎无静音（< 0.10s），不启用修正（避免误判）
    if speech_onset < 0.10:
        speech_onset = None

    # 2. Whisper 转写
    whisper_result = run_whisper(audio, sr, model_name)

    # 2.5 原文对齐（有原文 → 对齐到原文；无原文 → 以识别文字为准）
    alignment_stats = None
    original_text = text.strip() if text else None
    if original_text:
        aligned = align_whisper_to_original(whisper_result["words"], original_text)
        whisper_result["original_text"] = original_text
        whisper_result["alignment_stats"] = aligned["stats"]
        whisper_result["words"] = aligned["words"]
        alignment_stats = aligned["stats"]
    else:
        whisper_result["words"] = _wrap_whisper_as_aligned(whisper_result["words"])
        whisper_result["original_text"] = None
        whisper_result["alignment_stats"] = None

    # 2.6 修正首词位置：若 Whisper 把句首静音计入了首词的 start，
    #     用 energy-based speech_onset 替代偏早的 start，让波形图上的
    #     高亮位置和实际语音对齐，同时修正 duration。
    if speech_onset:
        words_list = whisper_result["words"]
        for w in words_list:
            if w.get("start") is not None and w.get("end") is not None:
                if speech_onset > w["start"] + 0.03:
                    w["start"] = round(speech_onset, 3)
                    w["duration"] = round(w["end"] - w["start"], 3)
                break  # 只修正第一个有有效时间戳的词

    # 3. 基频
    pitch = analyze_pitch(audio, sr)

    # 4. 节奏
    rhythm = analyze_rhythm(whisper_result, len(audio) / sr, ground_truth_text=text, speech_onset=speech_onset)
    rhythm["_words_detail"] = whisper_result["words"]
    rhythm["noticeable_gaps_count"] = len(rhythm["noticeable_gaps"])

    # 5. 母语者对比（如果有）
    comparison = None
    native_words_for_tiers = None
    if ref_path:
        native_audio, native_sr = load_audio(ref_path)
        if native_sr != sr:
            native_audio = librosa.resample(native_audio.astype(np.float64), orig_sr=native_sr, target_sr=sr)
        native_whisper = run_whisper(native_audio, sr, model_name)
        native_words_for_tiers = native_whisper["words"]  # 存下来给三线图用
        native_pitch = analyze_pitch(native_audio, sr)
        comparison = compare_with_native(whisper_result, native_whisper, pitch, native_pitch,
                                         ground_truth_text=text)

    # 6. 逐词诊断
    f0_arr = np.array(pitch.get("_f0_array", [])) if pitch else None
    f0_t = np.array(pitch.get("_f0_times", [])) if pitch else None
    word_diag = diagnose_per_word(whisper_result, rhythm, comparison, pitch, f0_arr, f0_t, speech_onset=speech_onset)

    # 6.3 CMU 音素级分析（重音位置 + 发音偏差定位）
    cmu_analysis = analyze_word_stress(whisper_result["words"], whisper_result)

    # 6.4 将 CMU 分析结果织入逐词诊断
    _weave_cmu_into_word_diag(word_diag, cmu_analysis, whisper_result["words"])

    # 6.5 模板/内容比例分析（DI/RL/SGD 专项）
    template_analysis = analyze_template_content_ratio(whisper_result["words"], question_type)

    # 6.6 为问题词生成发音片段（学生原音 + 字典音，均 base64 内嵌）
    import requests as req
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from urllib.parse import quote

    # 先收集需要处理的词
    problem_words = [w for w in word_diag
                     if w["severity"] in ("error", "warning") and not w.get("is_function_word")]

    # 学生原音（本地，快速；缺失词无时间戳则跳过）
    for w in problem_words:
        src_word = whisper_result["words"][w["index"]]
        if src_word.get("start") is not None and src_word.get("end") is not None:
            uri = _word_audio_uri(audio, sr, src_word["start"], src_word["end"])
            w["student_audio_uri"] = uri

    # 字典音（Edge TTS 并行生成，国内可访问）
    def _fetch_dict_audio(clean_word):
        try:
            mp3_bytes = _edge_tts(clean_word)
            if mp3_bytes and len(mp3_bytes) > 500:
                return f"data:audio/mpeg;base64,{base64.b64encode(mp3_bytes).decode()}"
        except Exception:
            pass
        return None

    if problem_words:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(_fetch_dict_audio, w["clean"]): w for w in problem_words}
            for f in as_completed(futures):
                w = futures[f]
                try:
                    result = f.result()
                    if result:
                        w["dict_audio_uri"] = result
                except Exception:
                    pass

    # 7. 映射诊断模板
    flags = map_to_diagnostic_template(rhythm, pitch, whisper_result, comparison, cmu_analysis)

    # 7.5 连读/失爆分析
    connected_speech = _analyze_connected_speech(whisper_result["words"])

    # 7.6 生成诊断波形图
    waveform_b64 = generate_diagnostic_waveform(
        audio, sr, whisper_result["words"], word_diag, rhythm,
        student_name=student_name, comparison=comparison
    )

    # 8. 生成 Markdown 报告
    md_content = generate_markdown_draft(audio_path, rhythm, pitch, flags, whisper_result, comparison, student_name, word_diag, connected_speech, waveform_b64, native_words_for_tiers, cmu_analysis, template_analysis, question_type)
    base = Path(audio_path).stem
    md_path = str(Path(audio_path).parent / f"{base}_诊断草稿.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 8. 保存 JSON
    json_path = str(Path(audio_path).parent / f"{base}_analysis.json")
    output = {
        "audio_path": audio_path,
        "model": model_name,
        "text": whisper_result.get("text", ""),
        "original_text": whisper_result.get("original_text"),
        "alignment_stats": whisper_result.get("alignment_stats"),
        "words": whisper_result["words"],
        "rhythm": {k: v for k, v in rhythm.items() if k != "_words_detail"},
        "pitch": {k: v for k, v in pitch.items() if k != "f0_values" and k != "sentences"},
        "comparison": comparison,
        "flags": flags,
        "cmu_analysis": cmu_analysis,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return {
        "markdown": md_content,
        "md_path": md_path,
        "json_path": json_path,
        "whisper_result": whisper_result,
        "rhythm": rhythm,
        "pitch": pitch,
        "comparison": comparison,
        "word_diag": word_diag,
        "flags": flags,
        "cmu_analysis": cmu_analysis,
        "template_analysis": template_analysis,
        "question_type": question_type,
        "audio_duration": len(audio) / sr,
        "waveform_b64": waveform_b64,
    }


def main():
    parser = argparse.ArgumentParser(description="PTE 口语诊断 AI 分析 v2")
    parser.add_argument("audio", help="学生录音路径")
    parser.add_argument("--ref", "--reference", help="母语者同文本录音（可选）", default=None)
    parser.add_argument("--text", help="原文文本（用于纠正语音识别偏差，使对比对齐到正确文本）", default=None)
    parser.add_argument("--model", help="语音识别模型", default="base",
                        choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--output-md", help="输出 Markdown 诊断报告路径", default=None)
    parser.add_argument("--output-json", help="输出 JSON 分析结果路径", default=None)
    parser.add_argument("--name", help="学员名（用于报告标题）", default="")
    args = parser.parse_args()

    print(f"🔊 加载学生录音: {args.audio}")
    audio, sr = load_audio(args.audio)
    print(f"   采样率={sr}, 时长={len(audio)/sr:.2f}s")

    print(f"\n🧠 语音识别模型 ({args.model}) 转写中...")
    whisper_result = run_whisper(audio, sr, args.model)
    print(f"   识别: \"{whisper_result['text']}\" ({len(whisper_result['words'])} 词)")

    if args.ref:
        print(f"\n👤 加载母语者录音: {args.ref}")

    # 委托给 run_analysis
    result = run_analysis(
        audio_path=args.audio,
        ref_path=args.ref,
        text=args.text,
        model_name=args.model,
        student_name=args.name,
    )

    # 对齐信息
    if args.text and result["comparison"] and result["comparison"].get("alignment_stats"):
        stats = result["comparison"]["alignment_stats"]
        print(f"   对齐结果: {stats['matched']} 词匹配, "
              f"学生缺 {stats['student_missing']}, "
              f"母语缺 {stats['native_missing']}, "
              f"双方缺 {stats['both_missing']}")

    # 控制台报告
    print_console_report(result["rhythm"], result["pitch"], result["flags"],
                         result["whisper_result"], result["comparison"], args.audio)

    print(f"📄 Markdown 报告已保存: {result['md_path']}")
    print(f"📊 JSON 结果已保存: {result['json_path']}")


if __name__ == "__main__":
    main()
