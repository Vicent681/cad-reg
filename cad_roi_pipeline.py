#!/usr/bin/env python3
"""Two-stage CAD analysis pipeline with ROI detection, cropping, and value extraction."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image

from multimodal_prompt import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    call_modelscope_endpoint,
)


try:
    RESAMPLING_LANCZOS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - Pillow<10 fallback
    RESAMPLING_LANCZOS = Image.LANCZOS

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
SAFE_SLUG_RE = re.compile(r"[^a-z0-9]+")

ROI_SYSTEM_PROMPT = (
    "你是一名CAD图纸ROI定位助手。你的职责是根据用户的问题锁定包含关键信息的区域，"
    "并仅输出严格的JSON数据，字段包括rois（列表）和reasoning（字符串）。"
)

ROI_PROMPT_TEMPLATE = """你将看到一张像素宽度为 {width_px}、高度为 {height_px} 的CAD图纸。
用户问题：{question}
你的任务是基于用户问题定位所有可能读取答案所需尺寸、标注或文字的区域。请找出最多 {max_rois} 个最相关的ROI，这些区域必须覆盖回答问题所需的关键信息。
每个ROI必须包含以下字段：
- id：例如 \"ROI-1\"、\"ROI-2\"。
- description：简短描述该区域包含的元素。
- bbox：像素级坐标，字段为 x、y、width、height（x、y 从整张图左上角开始，单位像素）。

输出格式必须是JSON，不要包含额外文本。例如：
{{
  "rois": [
    {{"id": "ROI-1", "description": "楼板标高与剖面尺寸", "bbox": {{"x": 820, "y": 540, "width": 320, "height": 210}}}}
  ],
  "reasoning": "说明为何选择这些ROI。"
}}
"""

VALUE_PROMPT_TEMPLATE = """你将看到若干个从同一张CAD图纸裁剪出的ROI。
用户问题：{question}

ROI列表：
{roi_lines}

请综合所有ROI内容，给出问题所需的专业数值，并说明依据。
输出格式：
{{
  "analysis": "先解释如何从ROI推导数值。",
  "measurements": [
    {{"name": "参数名称", "value": 数值或字符串, "unit": "单位", "roi_id": "ROI-1"}}
  ],
  "answer": "面向用户的最终结论。"
}}
如果图中没有足够信息，请明确说明。
"""


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    normalized: bool = False

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "BoundingBox | None":
        if not isinstance(raw, dict):
            return None
        lowered = {str(k).lower(): v for k, v in raw.items()}

        def _value(*keys: str) -> float | None:
            for key in keys:
                if key in lowered:
                    return _to_float(lowered[key])
            return None

        x = _value("x", "left")
        y = _value("y", "top")
        width = _value("width", "w")
        height = _value("height", "h")

        if x is None and {"x1", "left"} <= lowered.keys():
            x = _value("x1", "left")
        if y is None and {"y1", "top"} <= lowered.keys():
            y = _value("y1", "top")

        if (width is None or height is None) and {"x1", "x2"} <= lowered.keys():
            x1 = _value("x1", "left")
            x2 = _value("x2", "right")
            if x1 is not None and x2 is not None:
                x = x or x1
                width = x2 - x1
        if (width is None or height is None) and {"y1", "y2"} <= lowered.keys():
            y1 = _value("y1", "top")
            y2 = _value("y2", "bottom")
            if y1 is not None and y2 is not None:
                y = y or y1
                height = y2 - y1

        if width is None and {"right"} <= lowered.keys() and x is not None:
            right = _value("right")
            if right is not None:
                width = right - x
        if height is None and {"bottom"} <= lowered.keys() and y is not None:
            bottom = _value("bottom")
            if bottom is not None:
                height = bottom - y

        if None in (x, y, width, height):
            return None

        normalized = bool(lowered.get("normalized", False))
        if not normalized:
            normalized = (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            )
        return cls(x=float(x), y=float(y), width=float(width), height=float(height), normalized=normalized)

    def to_pixels(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        if self.normalized:
            return (
                self.x * image_width,
                self.y * image_height,
                self.width * image_width,
                self.height * image_height,
            )
        return self.x, self.y, self.width, self.height


@dataclass
class ROIResult:
    roi_id: str
    description: str
    bbox: BoundingBox
    pixel_bbox: Dict[str, int]
    crop_path: Path
    confidence: float | None = None
    detection_bbox: Dict[str, int] | None = None


@dataclass
class ImageWorkspace:
    source_path: Path
    original_image: Image.Image
    original_width: int
    original_height: int
    detection_path: Path
    detection_width: int
    detection_height: int
    scale_x: float
    scale_y: float


def prepare_image_workspace(image_path: Path, output_dir: Path, max_detection_dim: int) -> ImageWorkspace:
    original_image = Image.open(image_path)
    orig_w, orig_h = original_image.size
    detection_path = image_path
    det_w, det_h = orig_w, orig_h

    if max_detection_dim > 0 and max(orig_w, orig_h) > max_detection_dim:
        detection_copy = original_image.copy()
        detection_copy.thumbnail((max_detection_dim, max_detection_dim), RESAMPLING_LANCZOS)
        detection_path = output_dir / f"{image_path.stem}_det.png"
        detection_copy.save(detection_path)
        det_w, det_h = detection_copy.size
        detection_copy.close()

    scale_x = orig_w / det_w
    scale_y = orig_h / det_h
    return ImageWorkspace(
        source_path=image_path,
        original_image=original_image,
        original_width=orig_w,
        original_height=orig_h,
        detection_path=detection_path,
        detection_width=det_w,
        detection_height=det_h,
        scale_x=scale_x,
        scale_y=scale_y,
    )


def rescale_rois_to_original(
    rois: Sequence[ROIResult],
    detection_width: int,
    detection_height: int,
    scale_x: float,
    scale_y: float,
) -> None:
    if not rois:
        return
    if detection_width <= 0 or detection_height <= 0:
        return
    if scale_x == 1.0 and scale_y == 1.0:
        return
    for roi in rois:
        det_px_bbox = roi.bbox.to_pixels(detection_width, detection_height)
        roi.bbox = BoundingBox(
            x=det_px_bbox[0] * scale_x,
            y=det_px_bbox[1] * scale_y,
            width=det_px_bbox[2] * scale_x,
            height=det_px_bbox[3] * scale_y,
            normalized=False,
        )


def _tuple_to_int_bbox(values: tuple[float, float, float, float]) -> Dict[str, int]:
    x, y, width, height = values
    return {
        "x": int(round(x)),
        "y": int(round(y)),
        "width": int(round(width)),
        "height": int(round(height)),
    }


def bbox_to_pixel_dict(bbox: BoundingBox, image_width: int, image_height: int) -> Dict[str, int]:
    return _tuple_to_int_bbox(bbox.to_pixels(image_width, image_height))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        match = NUMBER_RE.search(cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
    return None


def _extract_json_block(text: str) -> Dict[str, Any]:
    candidates: List[str] = []
    for match in JSON_BLOCK_RE.finditer(text):
        candidates.append(match.group(1))
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("未能解析模型返回的ROI JSON。")


def _resolve_api_key(explicit: str | None) -> str:
    env_candidates = [
        explicit,
        os.getenv("MODELSCOPE_API_KEY"),
        os.getenv("MODELSCOPE_TOKEN"),
        os.getenv("OPENAI_API_KEY"),
    ]
    for candidate in env_candidates:
        if candidate and candidate.strip():
            return candidate.strip()
    raise ValueError("请通过 --api-key 或环境变量提供有效的 ModelScope API Key。")


def detect_rois(
    image_path: Path,
    image_width: int,
    image_height: int,
    question: str,
    max_rois: int,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[List[ROIResult], Dict[str, Any]]:
    prompt = ROI_PROMPT_TEMPLATE.format(
        width_px=image_width,
        height_px=image_height,
        question=question.strip(),
        max_rois=max_rois,
    )
    response_text = call_modelscope_endpoint(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=ROI_SYSTEM_PROMPT,
        prompt=prompt,
        image_paths=[image_path],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    payload = _extract_json_block(response_text)
    rois_raw = payload.get("rois") or []
    roi_results: List[ROIResult] = []

    for idx, item in enumerate(rois_raw, start=1):
        if not isinstance(item, dict):
            continue
        bbox = BoundingBox.from_raw(item.get("bbox") or item)
        if not bbox or bbox.width <= 0 or bbox.height <= 0:
            continue
        det_px_bbox = bbox.to_pixels(image_width, image_height)
        roi_id = str(item.get("id") or f"ROI-{idx}")
        description = str(item.get("description") or item.get("desc") or "ROI 区域")
        confidence = _to_float(item.get("confidence"))
        roi_results.append(
            ROIResult(
                roi_id=roi_id,
                description=description,
                bbox=bbox,
                pixel_bbox={},
                crop_path=image_path,
                confidence=confidence,
                detection_bbox=_tuple_to_int_bbox(det_px_bbox),
            )
        )
        if len(roi_results) >= max_rois:
            break

    return roi_results, payload


def _expand_and_clamp_bbox(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    margin_ratio: float,
    min_padding: float,
    min_width: float,
    min_height: float,
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    pad_x = max(min_padding, width * margin_ratio)
    pad_y = max(min_padding, height * margin_ratio)
    left = x - pad_x
    top = y - pad_y
    right = x + width + pad_x
    bottom = y + height + pad_y

    min_width = min(max(min_width, 1.0), float(image_width))
    min_height = min(max(min_height, 1.0), float(image_height))

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    half_width = max((right - left) / 2.0, min_width / 2.0)
    half_height = max((bottom - top) / 2.0, min_height / 2.0)

    left = center_x - half_width
    right = center_x + half_width
    top = center_y - half_height
    bottom = center_y + half_height

    # Clamp to image bounds while keeping requested size when possible
    if left < 0.0:
        right -= left
        left = 0.0
    if right > image_width:
        shift = right - image_width
        right = float(image_width)
        left = max(0.0, left - shift)
    if top < 0.0:
        bottom -= top
        top = 0.0
    if bottom > image_height:
        shift = bottom - image_height
        bottom = float(image_height)
        top = max(0.0, top - shift)

    if right <= left:
        right = min(float(image_width), left + 1.0)
    if bottom <= top:
        bottom = min(float(image_height), top + 1.0)

    return int(round(left)), int(round(top)), int(round(right)), int(round(bottom))


def slugify(text: str) -> str:
    slug = SAFE_SLUG_RE.sub("-", text.lower()).strip("-")
    if not slug:
        slug = "roi"
    return slug[:48]


def crop_rois(
    image: Image.Image,
    source_path: Path,
    roi_candidates: Sequence[ROIResult],
    output_dir: Path,
    margin_ratio: float,
    min_padding: float,
    min_width_ratio: float,
    min_height_ratio: float,
) -> tuple[List[ROIResult], Dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = image.size
    min_width = width * max(0.0, min_width_ratio)
    min_height = height * max(0.0, min_height_ratio)
    cropped: List[ROIResult] = []
    for idx, roi in enumerate(roi_candidates, start=1):
        px_bbox = roi.bbox.to_pixels(width, height)
        left, top, right, bottom = _expand_and_clamp_bbox(
            px_bbox,
            width,
            height,
            margin_ratio,
            min_padding,
            min_width,
            min_height,
        )
        if right - left <= 0 or bottom - top <= 0:
            continue
        crop = image.crop((left, top, right, bottom))
        slug = slugify(roi.description or roi.roi_id)
        crop_path = output_dir / f"{source_path.stem}_{roi.roi_id.lower()}_{slug}.png"
        crop.save(crop_path)
        roi.pixel_bbox = {
            "x": int(left),
            "y": int(top),
            "width": int(right - left),
            "height": int(bottom - top),
        }
        roi.crop_path = crop_path
        cropped.append(roi)

    if not cropped:
        fallback_path = output_dir / f"{source_path.stem}_full.png"
        image.save(fallback_path)
        full_bbox = BoundingBox(0.0, 0.0, float(width), float(height), normalized=False)
        fallback = ROIResult(
            roi_id="ROI-1",
            description="整张图",
            bbox=full_bbox,
            pixel_bbox={"x": 0, "y": 0, "width": width, "height": height},
            crop_path=fallback_path,
            detection_bbox={"x": 0, "y": 0, "width": width, "height": height},
        )
        cropped.append(fallback)

    return cropped, {"width": width, "height": height}


def _format_roi_lines(rois: Sequence[ROIResult]) -> str:
    lines = []
    for roi in rois:
        bbox = roi.pixel_bbox
        detail = (
            f"- {roi.roi_id}: {roi.description} "
            f"(x={bbox['x']}, y={bbox['y']}, width={bbox['width']}, height={bbox['height']})"
        )
        lines.append(detail)
    return "\n".join(lines)


def extract_values_from_rois(
    rois: Sequence[ROIResult],
    question: str,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    image_paths = [roi.crop_path for roi in rois]
    prompt = VALUE_PROMPT_TEMPLATE.format(
        question=question.strip(),
        roi_lines=_format_roi_lines(rois),
    )
    return call_modelscope_endpoint(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        prompt=prompt,
        image_paths=image_paths,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def save_metadata(
    output_dir: Path,
    source_image: Path,
    question: str,
    roi_payload: Dict[str, Any],
    rois: Sequence[ROIResult],
    final_answer: str,
    image_size: Dict[str, int],
) -> Path:
    summary = {
        "source_image": str(source_image),
        "question": question,
        "image_width": image_size.get("width"),
        "image_height": image_size.get("height"),
        "roi_detection_raw": roi_payload,
        "rois": [
            {
                "id": roi.roi_id,
                "description": roi.description,
                "bbox": {
                    "x": roi.pixel_bbox["x"],
                    "y": roi.pixel_bbox["y"],
                    "width": roi.pixel_bbox["width"],
                    "height": roi.pixel_bbox["height"],
                },
                "crop_path": str(roi.crop_path),
                "confidence": roi.confidence,
            }
            for roi in rois
        ],
        "answer": final_answer.strip(),
    }
    metadata_path = output_dir / "roi_summary.json"
    metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAD图纸ROI裁剪与数值识别流水线。",
    )
    parser.add_argument("image", type=Path, help="待分析的CAD图纸（PNG）。")
    parser.add_argument("--question", required=True, help="用户提出的问题或任务。")
    parser.add_argument("--output-dir", type=Path, default=Path("roi_outputs"), help="ROI裁剪与结果输出目录。")
    parser.add_argument("--api-key", help="ModelScope/OpenAI 兼容API key。")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="模型推理Base URL。")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="模型ID。默认读取环境变量 MODELSCOPE_MODEL，否则使用官方Qwen3 VL模型。",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="数值识别阶段的系统提示词。")
    parser.add_argument("--temperature", type=float, default=0.2, help="数值识别阶段的temperature。")
    parser.add_argument("--max-tokens", type=int, default=2048, help="数值识别阶段的最大输出token。")
    parser.add_argument("--roi-temperature", type=float, default=0.1, help="ROI检测阶段的temperature。")
    parser.add_argument("--roi-max-tokens", type=int, default=1024, help="ROI检测阶段的最大token。")
    parser.add_argument("--max-rois", type=int, default=3, help="最多裁剪的ROI数量。")
    parser.add_argument("--bbox-margin", type=float, default=0.08, help="ROI扩展边距占原ROI尺寸的比例。")
    parser.add_argument(
        "--bbox-min-padding",
        type=float,
        default=32.0,
        help="ROI扩展的最小像素padding，用于覆盖尺寸标注和文字。",
    )
    parser.add_argument(
        "--bbox-min-width-ratio",
        type=float,
        default=0.1,
        help="ROI裁剪后宽度相对于整张图的最小比例，用于避免过窄的裁剪区域。",
    )
    parser.add_argument(
        "--bbox-min-height-ratio",
        type=float,
        default=0.08,
        help="ROI裁剪后高度相对于整张图的最小比例，用于包含标注上下文。",
    )
    parser.add_argument(
        "--max-detection-size",
        type=int,
        default=2048,
        help="ROI检测阶段使用的图像最长边像素（大图会先等比例缩放，再送入模型；0 表示不缩放）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    image_path: Path = args.image
    if not image_path.exists():
        raise SystemExit(f"图纸文件不存在: {image_path}")
    if args.max_detection_size < 0:
        raise SystemExit("--max-detection-size 必须 >= 0。")
    if args.bbox_min_padding < 0:
        raise SystemExit("--bbox-min-padding 必须 >= 0。")
    if args.bbox_min_width_ratio < 0 or args.bbox_min_height_ratio < 0:
        raise SystemExit("--bbox-min-width-ratio 和 --bbox-min-height-ratio 必须 >= 0。")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        workspace = prepare_image_workspace(
            image_path=image_path,
            output_dir=output_dir,
            max_detection_dim=args.max_detection_size,
        )
    except OSError as exc:
        raise SystemExit(f"无法读取图纸（请确认为标准PNG）: {exc}") from exc

    api_key = _resolve_api_key(args.api_key)
    question = args.question.strip()
    if not question:
        raise SystemExit("请输入有效的问题。")

    try:
        print("1) 正在调用模型定位ROI ...")
        try:
            roi_candidates, roi_payload = detect_rois(
                image_path=workspace.detection_path,
                image_width=workspace.detection_width,
                image_height=workspace.detection_height,
                question=question,
                max_rois=args.max_rois,
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                temperature=args.roi_temperature,
                max_tokens=args.roi_max_tokens,
            )
            rescale_rois_to_original(
                rois=roi_candidates,
                detection_width=workspace.detection_width,
                detection_height=workspace.detection_height,
                scale_x=workspace.scale_x,
                scale_y=workspace.scale_y,
            )
        except Exception as exc:
            print(f"   [警告] ROI检测失败，将使用整图进行识别：{exc}")
            roi_candidates = []
            roi_payload = {"error": str(exc)}
        if roi_candidates:
            for roi in roi_candidates:
                det_bbox = roi.detection_bbox or {}
                mapped_bbox = bbox_to_pixel_dict(
                    roi.bbox,
                    workspace.original_width,
                    workspace.original_height,
                )
                print(f"   - {roi.roi_id}: {roi.description}")
                print(f"       模型ROI (检测图 {workspace.detection_width}x{workspace.detection_height}): {det_bbox}")
                print(f"       映射到原图 ({workspace.original_width}x{workspace.original_height}): {mapped_bbox}")
        else:
            print("   - 未检测到ROI，默认裁剪整张图。")

        print("2) 正在裁剪ROI ...")
        cropped_rois, image_size = crop_rois(
            image=workspace.original_image,
            source_path=workspace.source_path,
            roi_candidates=roi_candidates,
            output_dir=output_dir,
            margin_ratio=args.bbox_margin,
            min_padding=args.bbox_min_padding,
            min_width_ratio=args.bbox_min_width_ratio,
            min_height_ratio=args.bbox_min_height_ratio,
        )
        for roi in cropped_rois:
            bbox = roi.pixel_bbox
            print(f"   - {roi.roi_id}: 扩展后ROI {bbox} -> {roi.crop_path}")

        print("3) 正在识别ROI中的数值 ...")
        try:
            final_answer = extract_values_from_rois(
                rois=cropped_rois,
                question=question,
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                system_prompt=args.system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:
            raise SystemExit(f"数值识别阶段失败: {exc}") from exc
        print("\n=== ROI裁剪结果 ===")
        for roi in cropped_rois:
            bbox = roi.pixel_bbox
            print(
                f"{roi.roi_id}: {roi.description} "
                f"(x={bbox['x']}, y={bbox['y']}, width={bbox['width']}, height={bbox['height']}) -> {roi.crop_path}"
            )
        print("\n=== 数值识别输出 ===")
        print(final_answer.strip() or "[模型未返回结果]")

        metadata_path = save_metadata(
            output_dir=output_dir,
            source_image=image_path,
            question=question,
            roi_payload=roi_payload,
            rois=cropped_rois,
            final_answer=final_answer,
            image_size=image_size,
        )
        print(f"\n元数据已保存：{metadata_path}")
    finally:
        workspace.original_image.close()


if __name__ == "__main__":
    main()
