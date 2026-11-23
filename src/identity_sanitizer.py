#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frontier Model Identity Sanitizer & Third-Party Leak Purger
===========================================================
Detects, purges, and sanitizes third-party model references commonly found
in distilled datasets (OpenAI, ChatGPT, Claude, Anthropic, DeepSeek, Qwen,
Baichuan, ChatGLM, etc.).

Key Differentiators:
1. Strict Context-Aware Detection:
   - Purges first-person identity claims: "I am ChatGPT", "As an AI trained by OpenAI",
     "I am Claude", "As an AI developed by Anthropic", "I am DeepSeek", "I am Qwen".
   - Purges policy boilerplate: "According to OpenAI policies", "As an AI created by OpenAI I cannot".
   - PRESERVES valid third-person and factual context: "Claude Debussy", "anthropic principle",
     "import openai", "when was OpenAI founded", comparing API architectures.
2. Sanitization vs. Purging:
   - If a sample is purely a third-party identity declaration or refusal, it is PURGED (dropped).
   - If a sample is a high-value math/code solution with a canned preamble (e.g. "As an AI trained by OpenAI,
     here is the step-by-step solution..."), the toxic preamble is stripped while the mathematical reasoning is preserved.
"""

import re
from typing import Optional, Tuple, Dict, Any, List

# ─────────────────────────────────────────────────────────────────────────────
# REGEX PATTERNS FOR FORBIDDEN IDENTITY & BOILERPLATE CLAIMS
# ─────────────────────────────────────────────────────────────────────────────

# 1. First-person persona claims (Model claiming to be another entity)
FORBIDDEN_SELF_IDENTITY_REGEXES = [
    # Direct first-person model name claims
    re.compile(r"\b(?:i\s*am|i'm|as)\s+(?:chatgpt|gpt-?3\.?5|gpt-?4o?|gpt-?5|claude|qwen|deepseek|bard|gemini|llama|baichuan|chatglm|glm-4|kimi|moonshot|ernie\s*bot)\b", re.IGNORECASE),

    # Company attribution: trained / created / developed by <company>
    re.compile(r"\b(?:trained|created|developed|built|engineered)\s+by\s+(?:openai|anthropic|google|deepmind|meta|deepseek|alibaba|baidu|zhipu|moonshot|baichuan)\b", re.IGNORECASE),

    # AI entity developed by <company>
    re.compile(r"\b(?:an?\s+)?(?:ai|assistant|language\s+model|large\s+language\s+model)\s+(?:trained|created|developed|built|engineered)\s+by\s+(?:openai|anthropic|google|deepmind|meta|deepseek|alibaba|baidu|zhipu|moonshot|baichuan)\b", re.IGNORECASE),

    # Creators / developers at <company>
    re.compile(r"\bmy\s+(?:creators?|developers?|engineers?|team)\s+at\s+(?:openai|anthropic|google|deepmind|meta|deepseek|alibaba|baidu)\b", re.IGNORECASE),

    # Direct company trained me
    re.compile(r"\b(?:openai|anthropic|deepseek|alibaba|google)\s+trained\s+me\b", re.IGNORECASE),

    # Chinese model identity declarations (anchor-free for CJK characters)
    re.compile(r"我是(?:由)?深度求索(?:(?:开发|训练|研发)的)?", re.IGNORECASE),
    re.compile(r"深度求索(?:开发|研发|训练)的(?:人工智能|ai|助手)", re.IGNORECASE),
    re.compile(r"我是\s*deepseek", re.IGNORECASE),
    re.compile(r"我是(?:由)?(?:阿里|阿里巴巴|通义千问)(?:(?:开发|训练|研发)的)?", re.IGNORECASE),
    re.compile(r"阿里巴巴(?:开发|研发|训练)的(?:人工智能|ai|助手|语言模型)", re.IGNORECASE),
    re.compile(r"我是\s*通义千问", re.IGNORECASE),
    re.compile(r"我是\s*qwen", re.IGNORECASE),
    re.compile(r"我是(?:由)?(?:百川|智谱|chatglm|kimi|月之暗面|百度|文心一言)(?:(?:开发|训练|研发)的)?", re.IGNORECASE),

    # Claude exclusions (allow Claude Debussy, Claude Shannon, etc.)
    re.compile(r"\b(?:i\s*am|i'm)\s+claude\b(?!\s+(?:debussy|shannon|monet|levi-strauss|perrault))", re.IGNORECASE),
    re.compile(r"\bas\s+claude\b(?!\s+(?:debussy|shannon|monet|levi-strauss|perrault))", re.IGNORECASE),
    re.compile(r"\bconstitutional\s+ai\s+by\s+anthropic\b", re.IGNORECASE),
]

# 2. Policy / Canned Refusal Boilerplate
FORBIDDEN_POLICY_BOILERPLATE = [
    re.compile(r"\baccording\s+to\s+openai(?:'s)?\s+(?:content\s+)?(?:policy|policies|guidelines|use\s+case\s+policy)\b", re.IGNORECASE),
    re.compile(r"\bviolates?\s+openai(?:'s)?\s+(?:content\s+)?(?:policy|policies|safety\s+guidelines)\b", re.IGNORECASE),
    re.compile(r"\banthropic(?:'s)?\s+(?:acceptable\s+use\s+policy|safety\s+guidelines)\b", re.IGNORECASE),
    re.compile(r"\bas\s+a\s+responsible\s+ai\s+developed\s+by\s+openai\b", re.IGNORECASE),
    re.compile(r"\bmy\s+knowledge\s+cutoff\s+is\s+(?:september|october|november|december|january)\s+202[1-4]\b", re.IGNORECASE),
]

# 3. Preamble patterns that can be cleanly stripped from an otherwise good answer
STRIPPABLE_PREAMBLE_PATTERNS = [
    re.compile(r"^(?:hello!?\s*)?(?:as\s+(?:an?\s+)?(?:ai|artificial\s+intelligence)\s+(?:language\s+model\s+)?(?:trained|developed|created)\s+by\s+(?:openai|anthropic|google|deepseek|alibaba)[^.,\n]{0,200}[.,\n]\s*)", re.IGNORECASE),
    re.compile(r"^(?:hello!?\s*)?(?:(?:i\s*am|i'm)\s+(?:an?\s+)?(?:ai|assistant|language\s+model)\s+(?:trained|developed|created)\s+by\s+(?:openai|anthropic|google|deepseek|alibaba)[^.,\n]{0,200}[.,\n]\s*)", re.IGNORECASE),
    re.compile(r"^(?:as\s+(?:chatgpt|claude|qwen|deepseek)[^.,\n]{0,200}[.,\n]\s*)", re.IGNORECASE),
    re.compile(r"^(?:(?:i\s*am|i'm)\s+(?:chatgpt|claude|qwen|deepseek)[^.,\n]{0,200}[.,\n]\s*)", re.IGNORECASE),
]

# 4. System prompt leaks (e.g. from distilled synthetic conversations)
SYSTEM_PROMPT_LEAK_PATTERNS = [
    re.compile(r"\byou\s+are\s+chatgpt,\s+a\s+large\s+language\s+model\s+trained\s+by\s+openai\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+claude,\s+an\s+ai\s+assistant\s+created\s+by\s+anthropic\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+qwen,\s+created\s+by\s+alibaba\s+cloud\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+deepseek-?(?:v2|v3|r1|chat|coder)?\b", re.IGNORECASE),
]

# Fast Pre-Filter Keywords (O(1) substring check before running regexes)
TRIGGER_KEYWORDS = (
    "openai", "chatgpt", "gpt-3", "gpt-4", "gpt-5",
    "claude", "anthropic",
    "deepseek", "深度求索",
    "qwen", "tongyi", "通义千问", "阿里巴巴", "阿里",
    "baichuan", "百川", "chatglm", "智谱", "glm-4",
    "kimi", "moonshot", "月之暗面",
    "ernie", "文心一言",
    "gemini", "bard", "llama",
    "knowledge cutoff",
)


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def is_identity_contaminated(text: str) -> Tuple[bool, Optional[str]]:
    """
    Checks whether a text string contains unauthorized third-party model identity claims,
    policy boilerplates, or Chinese-distilled model self-declarations.
    
    Optimized with O(N) substring gating for blazing performance across gigabyte corpora.
    """
    if not text:
        return False, None

    text_lower = text.lower()

    # Fast gate: if no trigger keyword exists, immediately return False
    has_trigger = False
    for kw in TRIGGER_KEYWORDS:
        if kw in text_lower:
            has_trigger = True
            break

    if not has_trigger:
        return False, None

    # 1. Check self-identity claims
    for regex in FORBIDDEN_SELF_IDENTITY_REGEXES:
        match = regex.search(text)
        if match:
            return True, f"Forbidden identity claim: '{match.group(0)}'"

    # 2. Check policy boilerplate
    for regex in FORBIDDEN_POLICY_BOILERPLATE:
        match = regex.search(text)
        if match:
            return True, f"Third-party policy boilerplate: '{match.group(0)}'"

    # 3. Check system prompt leaks
    for regex in SYSTEM_PROMPT_LEAK_PATTERNS:
        match = regex.search(text)
        if match:
            return True, f"System prompt leak: '{match.group(0)}'"

    return False, None


def sanitize_text(text: str) -> Tuple[str, bool]:
    """
    Attempts to clean strippable preambles or boilerplate from an otherwise valid response.
    
    Returns:
        (sanitized_text: str, modified: bool)
    """
    if not text:
        return text, False

    original = text
    cleaned = text

    # Strip conversational toxic preambles
    for pat in STRIPPABLE_PREAMBLE_PATTERNS:
        cleaned = pat.sub("", cleaned).strip()

    # If the text was modified and now no longer has contamination, return it
    if cleaned != original:
        is_bad, _ = is_identity_contaminated(cleaned)
        if not is_bad:
            return cleaned, True

    return original, False


def sanitize_conversation_turn(
    prompt: str,
    response: str,
    allow_adversarial_attribution: bool = True
) -> Optional[Dict[str, str]]:
    """
    Sanitizes or purges a prompt-response interaction:
    
    1. If `allow_adversarial_attribution` is True:
       Allows samples where the user *asks* about OpenAI/Claude/etc., and the model
       correctly *rejects* the claim (e.g. from anti-sycophancy or Doty identity sets).
    2. If the response claims the model IS OpenAI/Claude/Qwen/DeepSeek:
       - Tries to strip preamble if substantive reasoning follows.
       - If the response is purely identity boilerplate, returns None (PURGED).
    3. If prompt has system prompt leaks, returns None (PURGED).
    """
    # Check prompt for system prompt echoes
    for regex in SYSTEM_PROMPT_LEAK_PATTERNS:
        if regex.search(prompt):
            return None  # Purge tainted prompt

    # Check response for identity contamination
    is_bad, reason = is_identity_contaminated(response)

    if not is_bad:
        return {"prompt": prompt, "response": response}

    # If it's an adversarial defense where the model explicitly rejects the third-party identity:
    if allow_adversarial_attribution:
        # Check if the response contains rejection phrases:
        rejection_indicators = [
            "i am not", "i'm not", "not developed by", "not created by",
            "not chatgpt", "not claude", "no, that is incorrect", "no, i am",
            "neither", "independent model", "developed by ajit kumar", "i am doty"
        ]
        resp_lower = response.lower()
        if any(ind in resp_lower for ind in rejection_indicators) and "ajit kumar" in resp_lower:
            # Valid adversarial defense sample! Preserve it!
            return {"prompt": prompt, "response": response}

    # Attempt to sanitize by stripping canned preamble
    cleaned_resp, modified = sanitize_text(response)
    if modified:
        is_still_bad, _ = is_identity_contaminated(cleaned_resp)
        if not is_still_bad and len(cleaned_resp.strip()) > 5:
            return {"prompt": prompt, "response": cleaned_resp.strip()}

    # Otherwise, this sample is genuinely contaminated with third-party identity -> PURGE IT
    return None


def purge_dataset(
    input_records: List[Dict[str, Any]],
    text_key: str = "formatted_output",
    prompt_key: str = "formatted_input",
    allow_adversarial: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Processes a list of records, purging contaminated third-party model entries
    and sanitizing salvageable reasoning traces across both flat (prompt/response)
    and multi-turn (messages / conversations) schemas.
    """
    clean_records = []
    stats = {
        "total_in": len(input_records),
        "retained": 0,
        "sanitized": 0,
        "purged_identity_leak": 0,
    }

    for item in input_records:
        # Multi-turn ShareGPT / OpenAI 'messages' schema
        if "messages" in item and isinstance(item["messages"], list):
            purged = False
            modified_any = False
            user_prompts = [m.get("content", "") for m in item["messages"] if m.get("role") in ("user", "human")]
            full_user_prompt = "\n".join(user_prompts)

            for m in item["messages"]:
                role = m.get("role")
                content = m.get("content", "")
                if role in ("system",):
                    for regex in SYSTEM_PROMPT_LEAK_PATTERNS:
                        if regex.search(content):
                            purged = True
                            break
                    if purged:
                        break
                elif role in ("assistant", "gpt", "model"):
                    res = sanitize_conversation_turn(full_user_prompt, content, allow_adversarial_attribution=allow_adversarial)
                    if res is None:
                        purged = True
                        break
                    if res["response"] != content:
                        m["content"] = res["response"]
                        modified_any = True

            if purged:
                stats["purged_identity_leak"] += 1
                continue
            if modified_any:
                stats["sanitized"] += 1
            clean_records.append(item)
            stats["retained"] += 1
            continue

        # Multi-turn ShareGPT 'conversations' schema
        elif "conversations" in item and isinstance(item["conversations"], list):
            purged = False
            modified_any = False
            user_prompts = [m.get("value", "") for m in item["conversations"] if m.get("from") in ("user", "human")]
            full_user_prompt = "\n".join(user_prompts)

            for m in item["conversations"]:
                sender = m.get("from")
                content = m.get("value", "")
                if sender in ("system",):
                    for regex in SYSTEM_PROMPT_LEAK_PATTERNS:
                        if regex.search(content):
                            purged = True
                            break
                    if purged:
                        break
                elif sender in ("gpt", "assistant", "model"):
                    res = sanitize_conversation_turn(full_user_prompt, content, allow_adversarial_attribution=allow_adversarial)
                    if res is None:
                        purged = True
                        break
                    if res["response"] != content:
                        m["value"] = res["response"]
                        modified_any = True

            if purged:
                stats["purged_identity_leak"] += 1
                continue
            if modified_any:
                stats["sanitized"] += 1
            clean_records.append(item)
            stats["retained"] += 1
            continue

        # Flat dictionary schema
        prompt = item.get(prompt_key) or item.get("prompt", "")
        response = item.get(text_key) or item.get("response") or item.get("answer", "")

        res = sanitize_conversation_turn(prompt, response, allow_adversarial_attribution=allow_adversarial)
        if res is None:
            stats["purged_identity_leak"] += 1
            continue

        if res["response"] != response:
            stats["sanitized"] += 1
            # Update item with cleaned response
            if text_key in item:
                item[text_key] = res["response"]
            if "response" in item:
                item["response"] = res["response"]
            if "answer" in item:
                item["answer"] = res["response"]

        clean_records.append(item)
        stats["retained"] += 1

    return clean_records, stats
