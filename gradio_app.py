#!/usr/bin/env python3
"""Gradio front-end for the multimodal CAD analysis helper."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence

import gradio as gr

from multimodal_prompt import stream_modelscope_endpoint


DEFAULT_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"
DEFAULT_BASE_URL = os.getenv("MODELSCOPE_BASE_URL") or "https://api-inference.modelscope.cn/v1"
DEFAULT_SYSTEM_PROMPT = "你是一个CAD工程图纸识别的助手，你的任务是分析给定的图纸，给出专业的数值。"
SCROLL_CSS = """
.scroll-output textarea {
    min-height: 320px !important;
    max-height: 540px !important;
    overflow-y: auto !important;
    white-space: pre-wrap !important;
}
"""


def _resolve_file_paths(files: Sequence[object]) -> List[Path]:
    paths: List[Path] = []
    for file in files:
        if file is None:
            continue
        if isinstance(file, str):
            paths.append(Path(file))
        elif hasattr(file, "name"):
            paths.append(Path(file.name))
        elif isinstance(file, dict) and "name" in file:
            paths.append(Path(str(file["name"])))
    return paths


def _default_api_key() -> str | None:
    return (
        os.getenv("MODELSCOPE_API_KEY")
        or os.getenv("MODELSCOPE_TOKEN")
        or os.getenv("OPENAI_API_KEY")
    )


def analyze_cad_images(
    files: Sequence[object],
    prompt: str,
    extra_prompt: str,
    system_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
):
    if not files:
        yield "请至少上传一张图纸。"
        return
    paths = _resolve_file_paths(files)
    if not paths:
        yield "未能解析上传的图纸文件。"
        return
    missing = [p for p in paths if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        yield f"以下文件不存在或不可用: {missing_str}"
        return

    final_prompt = (prompt or "").strip()
    if extra_prompt and extra_prompt.strip():
        final_prompt = f"{final_prompt}\n\n补充说明：{extra_prompt.strip()}"
    if not final_prompt:
        yield "请输入提示词。"
        return

    resolved_key = api_key.strip() or _default_api_key()
    if not resolved_key:
        yield "请提供有效的 ModelScope API Key。"
        return

    aggregated = ""
    try:
        for delta in stream_modelscope_endpoint(
            base_url=base_url.strip() or DEFAULT_BASE_URL,
            api_key=resolved_key,
            model=model.strip() or DEFAULT_MODEL,
            system_prompt=system_prompt.strip() or DEFAULT_SYSTEM_PROMPT,
            prompt=final_prompt,
            image_paths=paths,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        ):
            aggregated += delta
            yield aggregated
    except Exception as exc:  # pragma: no cover - surface runtime errors in UI
        if aggregated:
            yield aggregated + f"\n\n[错误] {exc}"
        else:
            yield f"推理失败: {exc}"
        return

    if not aggregated:
        yield "模型未返回有效文本，请重试。"


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="CAD 图纸识别助手", css=SCROLL_CSS) as demo:
        gr.Markdown("## CAD 工程图纸识别助手\n上传多张 CAD 图纸并输入要分析的问题。")

        with gr.Row():
            file_input = gr.File(
                label="CAD 图纸 (可多张)",
                file_types=["image"],
                file_count="multiple",
                type="file",
            )
            output_box = gr.Textbox(
                label="分析结果",
                lines=16,
                interactive=False,
                elem_classes=["scroll-output"],
            )

        prompt_box = gr.Textbox(
            label="用户提示词",
            placeholder="例如：综合分析两张楼层图的差异并给出关键尺寸。",
            lines=3,
        )

        extra_prompt_box = gr.Textbox(
            label="补充提示词（可选）",
            placeholder="例如：请重点关注结构标高和楼层关系。",
            lines=2,
        )

        system_box = gr.Textbox(
            label="系统提示词",
            value=DEFAULT_SYSTEM_PROMPT,
            lines=2,
        )

        with gr.Accordion("高级设置", open=False):
            model_box = gr.Textbox(label="模型 ID", value=DEFAULT_MODEL)
            base_url_box = gr.Textbox(label="Base URL", value=DEFAULT_BASE_URL)
            api_key_box = gr.Textbox(
                label="ModelScope API Key",
                value=_default_api_key() or "",
                type="password",
            )
            temperature_slider = gr.Slider(
                label="Temperature",
                value=0.2,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
            )
            max_tokens_slider = gr.Slider(
                label="Max Tokens",
                value=2048,
                minimum=256,
                maximum=4096,
                step=128,
            )

        run_button = gr.Button("分析图纸", variant="primary")
        run_button.click(
            analyze_cad_images,
            inputs=[
                file_input,
                prompt_box,
                extra_prompt_box,
                system_box,
                model_box,
                base_url_box,
                api_key_box,
                temperature_slider,
                max_tokens_slider,
            ],
            outputs=output_box,
        )
    return demo


def _parse_bool_env(value: str | None) -> tuple[bool, bool]:
    if value is None:
        return False, False
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True, True
    if lowered in {"0", "false", "no", "off"}:
        return True, False
    return False, False


def launch_app() -> None:
    demo = build_interface().queue()
    share_env = os.getenv("GRADIO_SHARE")
    share_explicit, share_value = _parse_bool_env(share_env)
    server_name = os.getenv("GRADIO_SERVER_NAME")
    server_port = os.getenv("GRADIO_SERVER_PORT")
    port_value = int(server_port) if server_port and server_port.isdigit() else None

    launch_kwargs = {
        "share": share_value if share_explicit else False,
        "server_name": server_name,
        "server_port": port_value,
        "show_api": False,
    }

    try:
        demo.launch(**launch_kwargs)
    except ValueError as exc:
        if "share=True" in str(exc) and not share_explicit:
            demo.launch(
                share=True,
                server_name=server_name,
                server_port=port_value,
                show_api=False,
            )
        else:
            raise


if __name__ == "__main__":
    launch_app()
