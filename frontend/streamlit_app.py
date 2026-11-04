"""Streamlit frontend playground for the RouteWise LLM Gateway."""

import os
from typing import Any

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("ROUTEWISE_API_URL", "http://127.0.0.1:8000").rstrip("/")
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# Matches MAX_PROMPT_LENGTH in app/schemas.py
MAX_PROMPT_LENGTH = 4000

# Slightly above backend REQUEST_TIMEOUT_SECONDS (60s) so backend 504s arrive cleanly
CLIENT_TIMEOUT_SECONDS = 65

MODE_CONFIG: dict[str, dict[str, str]] = {
    "auto": {
        "icon": "🔀",
        "title": "Auto — Semantic + Rule-Based Routing",
        "short_label": "Auto",
        "model_hint": "Dynamic (Flash-Lite or Flash)",
        "description": (
            "Embeds the prompt to classify the task category and applies contextual "
            "complexity rules to select between Gemini Flash-Lite and Gemini Flash."
        ),
    },
    "fast": {
        "icon": "⚡",
        "title": "Fast — Gemini Flash-Lite",
        "short_label": "Fast",
        "model_hint": "gemini/gemini-3.5-flash-lite",
        "description": (
            "Bypasses semantic classification and routes directly to the cheap tier "
            "for low-latency everyday responses."
        ),
    },
    "quality": {
        "icon": "✦",
        "title": "Quality — Gemini Flash",
        "short_label": "Quality",
        "model_hint": "gemini/gemini-3.6-flash",
        "description": (
            "Bypasses semantic classification and routes directly to the strong tier "
            "for demanding reasoning, coding, and systems tasks."
        ),
    },
}

ROUTING_METHOD_LABELS: dict[str, str] = {
    "semantic_plus_rules": "Auto (semantic_plus_rules)",
    "manual_fast": "Fast (manual_fast)",
    "manual_quality": "Quality (manual_quality)",
}


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def inject_custom_css() -> None:
    """Apply restrained dark-mode developer tool styling with lilac accents."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1080px;
        }
        .rw-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 1rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid #262736;
        }
        .rw-logo {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #F3F4F6;
            margin: 0;
        }
        .rw-logo span {
            color: #A78BFA;
        }
        .rw-subtitle {
            font-size: 0.92rem;
            color: #9CA3AF;
            margin-top: 0.15rem;
        }
        .rw-status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 500;
            border: 1px solid #2D2E3F;
            background-color: #161722;
        }
        .rw-dot-online {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #34D399;
            display: inline-block;
        }
        .rw-dot-offline {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #F87171;
            display: inline-block;
        }
        .rw-mode-card {
            background-color: #161722;
            border: 1px solid #262736;
            border-radius: 8px;
            padding: 1rem 1.15rem;
            min-height: 175px;
            margin-bottom: 0.5rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .rw-mode-card-selected {
            background-color: #1B192E;
            border: 1.5px solid #A78BFA;
            box-shadow: 0 0 0 1px rgba(167, 139, 250, 0.2);
        }
        .rw-mode-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 0.5rem;
            margin-bottom: 0.4rem;
        }
        .rw-mode-title {
            font-size: 0.96rem;
            font-weight: 600;
            color: #F3F4F6;
            line-height: 1.3;
        }
        .rw-mode-icon {
            color: #A78BFA;
            margin-right: 0.35rem;
        }
        .rw-mode-pill {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            white-space: nowrap;
            color: #9CA3AF;
            border: 1px solid #2D2E3F;
            background-color: #12131C;
        }
        .rw-mode-pill-selected {
            color: #EDE9FE;
            border-color: #A78BFA;
            background-color: rgba(167, 139, 250, 0.18);
        }
        .rw-mode-model {
            font-size: 0.78rem;
            font-family: monospace;
            color: #A78BFA;
            margin-bottom: 0.5rem;
        }
        .rw-mode-desc {
            font-size: 0.85rem;
            color: #D1D5DB;
            line-height: 1.45;
        }
        .rw-card {
            background-color: #161722;
            border: 1px solid #262736;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
        }
        .rw-card-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #9CA3AF;
            margin-bottom: 0.25rem;
        }
        .rw-card-value {
            font-size: 0.95rem;
            color: #F3F4F6;
            font-weight: 500;
            word-break: break-word;
        }
        .rw-accent {
            color: #A78BFA;
        }
        .rw-char-counter {
            font-size: 0.82rem;
            color: #9CA3AF;
            text-align: right;
            margin-top: -0.4rem;
            margin-bottom: 0.75rem;
        }
        .rw-footer {
            margin-top: 3.5rem;
            padding-top: 1.25rem;
            border-top: 1px solid #262736;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.82rem;
            color: #9CA3AF;
        }
        .rw-footer-brand {
            font-weight: 600;
            color: #D1D5DB;
        }
        .rw-footer-brand span {
            color: #A78BFA;
        }
        .rw-footer-stack {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.45rem;
        }
        .rw-footer-sep {
            color: #A78BFA;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Backend Communication Helpers
# ---------------------------------------------------------------------------

def check_backend_health() -> bool:
    """Query GET /health to verify if the FastAPI server is reachable."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=2.5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "ok"
        return False
    except requests.RequestException:
        return False


def send_chat_request(prompt: str, mode: str) -> dict[str, Any]:
    """Send POST /chat to FastAPI and return a structured result dictionary."""
    payload = {"prompt": prompt, "mode": mode}

    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json=payload,
            timeout=CLIENT_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "error_title": "Backend Unavailable",
            "error_message": (
                f"Could not connect to the RouteWise API at {API_BASE_URL}. "
                "Verify that the FastAPI server is running."
            ),
            "request_id": None,
        }
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "error_title": "Request Timed Out",
            "error_message": (
                f"The request exceeded the {CLIENT_TIMEOUT_SECONDS}-second client timeout."
            ),
            "request_id": None,
        }
    except requests.exceptions.RequestException as exc:
        return {
            "ok": False,
            "error_title": "Network Error",
            "error_message": f"A network error occurred while contacting the gateway ({type(exc).__name__}).",
            "request_id": None,
        }

    request_id = response.headers.get("X-Request-ID")

    if response.status_code == 200:
        try:
            body = response.json()
        except ValueError:
            return {
                "ok": False,
                "error_title": "Invalid Server Response",
                "error_message": "The server returned HTTP 200, but the response body was not valid JSON.",
                "request_id": request_id,
            }

        required_fields = {"answer", "model", "model_tier", "routing_method", "reason", "latency_ms"}
        if not isinstance(body, dict) or not required_fields.issubset(body.keys()):
            return {
                "ok": False,
                "error_title": "Unexpected Response Schema",
                "error_message": "The server response did not match the expected ChatResponse contract.",
                "request_id": request_id,
            }

        return {
            "ok": True,
            "data": body,
            "submitted_mode": mode,
            "request_id": request_id,
        }

    # Handle structured HTTP error codes safely
    detail_msg = None
    try:
        err_json = response.json()
        if isinstance(err_json, dict) and isinstance(err_json.get("detail"), str):
            detail_msg = err_json["detail"]
    except ValueError:
        pass

    status = response.status_code
    if status == 422:
        return {
            "ok": False,
            "error_title": "Validation Error (HTTP 422)",
            "error_message": (
                "The prompt or mode failed validation. Ensure the prompt is between "
                f"1 and {MAX_PROMPT_LENGTH:,} non-whitespace characters."
            ),
            "request_id": request_id,
        }
    if status == 429:
        return {
            "ok": False,
            "error_title": "Rate Limit Reached (HTTP 429)",
            "error_message": detail_msg or "The model provider's rate limit or quota was reached. Try again later.",
            "request_id": request_id,
        }
    if status == 503:
        return {
            "ok": False,
            "error_title": "Service Unavailable (HTTP 503)",
            "error_message": detail_msg or "A required routing or provider credential service is unavailable.",
            "request_id": request_id,
        }
    if status == 504:
        return {
            "ok": False,
            "error_title": "Provider Timeout (HTTP 504)",
            "error_message": detail_msg or "The upstream model provider did not respond in time.",
            "request_id": request_id,
        }
    if status in (500, 502):
        return {
            "ok": False,
            "error_title": f"Gateway Error (HTTP {status})",
            "error_message": detail_msg or "The gateway encountered an error while processing the request.",
            "request_id": request_id,
        }

    return {
        "ok": False,
        "error_title": f"Unexpected Error (HTTP {status})",
        "error_message": detail_msg or "An unexpected HTTP status code was returned by the backend.",
        "request_id": request_id,
    }


def infer_mode_label(routing_method: str, submitted_mode: str) -> str:
    """Infer the selected operating mode from routing_method and UI selection."""
    if routing_method in ROUTING_METHOD_LABELS:
        return ROUTING_METHOD_LABELS[routing_method]
    mode_info = MODE_CONFIG.get(submitted_mode)
    if mode_info:
        return mode_info["short_label"]
    return "Unknown"


def select_mode(mode_key: str) -> None:
    """Update the selected operating mode in session state without calling the API."""
    st.session_state.selected_mode = mode_key


# ---------------------------------------------------------------------------
# Main Application UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="RouteWise — Intelligent LLM Gateway",
        page_icon="⚡",
        layout="wide",
    )
    inject_custom_css()

    # Initialise session state so reruns preserve the last response and selected mode
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "selected_mode" not in st.session_state:
        st.session_state.selected_mode = "auto"

    # 1. Header & Backend Status
    is_online = check_backend_health()
    status_dot = "rw-dot-online" if is_online else "rw-dot-offline"
    status_text = "Backend Connected" if is_online else "Backend Offline"

    st.markdown(
        f"""
        <div class="rw-header">
            <div>
                <div class="rw-logo">Route<span>Wise</span></div>
                <div class="rw-subtitle">Intelligent LLM Gateway</div>
            </div>
            <div class="rw-status-badge">
                <span class="{status_dot}"></span>
                <span>{status_text} ({API_BASE_URL})</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Mode Selection Cards
    st.subheader("Operating Mode")
    mode_order = ["fast", "quality", "auto"]
    mode_cols = st.columns(3)

    for col, mode_key in zip(mode_cols, mode_order):
        info = MODE_CONFIG[mode_key]
        is_selected = st.session_state.selected_mode == mode_key
        card_class = "rw-mode-card rw-mode-card-selected" if is_selected else "rw-mode-card"
        pill_class = "rw-mode-pill rw-mode-pill-selected" if is_selected else "rw-mode-pill"
        pill_text = "Selected" if is_selected else "Mode"
        btn_label = f"✓ {info['short_label']} Active" if is_selected else f"Select {info['short_label']}"
        btn_type = "primary" if is_selected else "secondary"

        with col:
            st.markdown(
                f"""
                <div class="{card_class}">
                    <div>
                        <div class="rw-mode-header">
                            <div class="rw-mode-title">
                                <span class="rw-mode-icon">{info['icon']}</span>{info['title']}
                            </div>
                            <span class="{pill_class}">{pill_text}</span>
                        </div>
                        <div class="rw-mode-model">{info['model_hint']}</div>
                        <div class="rw-mode-desc">{info['description']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                btn_label,
                key=f"select_mode_{mode_key}",
                type=btn_type,
                use_container_width=True,
                on_click=select_mode,
                args=(mode_key,),
            )

    selected_mode: str = st.session_state.selected_mode
    active_mode_info = MODE_CONFIG[selected_mode]

    # 3. Prompt Input
    st.subheader("Prompt")
    prompt_text = st.text_area(
        label="Enter your prompt",
        height=150,
        max_chars=MAX_PROMPT_LENGTH,
        placeholder=(
            "Enter a task or question (e.g., 'Design a fault-tolerant distributed webhook "
            "receiver and analyse the latency trade-offs' or 'Explain how binary search works')..."
        ),
        label_visibility="collapsed",
    )

    stripped_len = len(prompt_text.strip())
    st.markdown(
        f'<div class="rw-char-counter">{stripped_len:,} / {MAX_PROMPT_LENGTH:,} characters</div>',
        unsafe_allow_html=True,
    )

    col_submit, col_clear, _ = st.columns([1.4, 1, 5.6])
    with col_submit:
        submit_clicked = st.button("Submit Prompt", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear Output", use_container_width=True):
            st.session_state.last_result = None
            st.rerun()

    # Dispatch only when the user explicitly clicks Submit Prompt
    if submit_clicked:
        cleaned_prompt = prompt_text.strip()
        if not cleaned_prompt:
            st.warning("Please enter a non-empty prompt before submitting.")
        elif len(cleaned_prompt) > MAX_PROMPT_LENGTH:
            st.warning(f"Prompt exceeds the maximum length of {MAX_PROMPT_LENGTH:,} characters.")
        else:
            with st.spinner(f"Dispatching request in {active_mode_info['short_label']} mode..."):
                st.session_state.last_result = send_chat_request(cleaned_prompt, selected_mode)

    # 4. Response & Telemetry Panel
    result = st.session_state.last_result
    if result is not None:
        st.divider()

        if not result["ok"]:
            st.error(f"**{result['error_title']}**\n\n{result['error_message']}")
            if result.get("request_id"):
                st.caption(f"Request ID (`X-Request-ID`): `{result['request_id']}`")
        else:
            data: dict[str, Any] = result["data"]
            request_id: str = result.get("request_id") or "N/A"
            submitted_mode: str = result.get("submitted_mode", "auto")

            # Answer Display
            st.subheader("Generated Answer")
            with st.container(border=True):
                st.markdown(data["answer"])

            # Performance Metrics Row
            st.subheader("Performance Metrics")
            latency_sec = f"{data['latency_ms'] / 1000:.2f} s"
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latency", latency_sec)
            m2.metric("Prompt Tokens", f"{prompt_tokens:,}" if prompt_tokens is not None else "N/A")
            m3.metric("Completion Tokens", f"{completion_tokens:,}" if completion_tokens is not None else "N/A")
            m4.metric("Total Tokens", f"{total_tokens:,}" if total_tokens is not None else "N/A")

            # Routing & Observability Metadata Panel
            st.subheader("Routing & Gateway Metadata")
            inferred_mode = infer_mode_label(data["routing_method"], submitted_mode)
            task_category = data.get("task_category")
            similarity_score = data.get("similarity_score")

            category_display = task_category if task_category is not None else "Bypassed (Manual Mode)"
            similarity_display = (
                f"{similarity_score:.3f}" if isinstance(similarity_score, (int, float)) else "N/A"
            )

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(
                    f"""
                    <div class="rw-card">
                        <div class="rw-card-label">Selected Model</div>
                        <div class="rw-card-value rw-accent">{data['model']}</div>
                    </div>
                    <div class="rw-card">
                        <div class="rw-card-label">Task Category</div>
                        <div class="rw-card-value">{category_display}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f"""
                    <div class="rw-card">
                        <div class="rw-card-label">Model Tier</div>
                        <div class="rw-card-value">{data['model_tier']}</div>
                    </div>
                    <div class="rw-card">
                        <div class="rw-card-label">Similarity Score</div>
                        <div class="rw-card-value">{similarity_display}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with r3:
                st.markdown(
                    f"""
                    <div class="rw-card">
                        <div class="rw-card-label">Mode / Routing Method</div>
                        <div class="rw-card-value">{inferred_mode}</div>
                    </div>
                    <div class="rw-card">
                        <div class="rw-card-label">X-Request-ID</div>
                        <div class="rw-card-value" style="font-family: monospace; font-size: 0.85rem;">{request_id}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div class="rw-card">
                    <div class="rw-card-label">Routing Explanation</div>
                    <div class="rw-card-value" style="font-weight: 400;">{data['reason']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 5. Footer
    st.markdown(
        """
        <div class="rw-footer">
            <div class="rw-footer-brand">
                Route<span>Wise</span> — Intelligent LLM Gateway
            </div>
            <div class="rw-footer-stack">
                <span>Python</span>
                <span class="rw-footer-sep">•</span>
                <span>Streamlit</span>
                <span class="rw-footer-sep">•</span>
                <span>FastAPI</span>
                <span class="rw-footer-sep">•</span>
                <span>LiteLLM</span>
                <span class="rw-footer-sep">•</span>
                <span>Gemini API</span>
                <span class="rw-footer-sep">•</span>
                <span>SQLite</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()