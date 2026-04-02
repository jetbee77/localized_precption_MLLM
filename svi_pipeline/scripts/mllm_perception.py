from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from PIL import Image
import io

from common import ensure_dir, load_config, safe_json_loads


def encode_image(image_path: Path) -> str:
    # Re-encode to PNG to ensure valid image bytes
    img = Image.open(image_path)
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_prompt(version: str) -> str:
    base = (
        "You are a resident very familiar with Beijing. Please rate your subjective perception of the street "
        "environment based on the image.\n"
        "The image is a stitched panorama of the same location from four directions.\n\n"
        "Output JSON with the following four fields. Score range is -1.00 to 1.00, keep two decimals:\n"
        "CQ: Cultural quality (traditional character, cultural atmosphere, historical symbols)\n"
        "AQ: Aesthetic quality (overall beauty, coherence, visual comfort)\n"
        "HQ: Harmony quality (sense of harmony, visual consistent, unified)\n"
        "VQ: Vitality quality (spatial vitality, participation, activity atmosphere)\n\n"
        "Requirements:\n"
        "1 Output JSON only, no explanations.\n"
        "2 If unsure, give the most reasonable estimate.\n"
        "Example: {\"CQ\":0.62,\"AQ\":0.55,\"HQ\":0.48,\"VQ\":0.31}\n"
    )

    if version == "local_prompt":
        return (
            base
            + "\nAdditional context: common elements of Beijing hutongs/lanes include gray-brick courtyard walls, "
            "traditional gateways, eaves, signboards, a narrow street scale, continuous street frontage, "
            "tree shade, and along-street daily-life scenes. Please pay special attention to these cultural "
            "and historical cues.\n"
        )

    if version == "anchor_prompt":
        return (
            base
            + "\nAnchor reference:\n"
            + "High: rich traditional elements, continuous street frontage, strong historical atmosphere, "
            "coherent and comfortable space.\n"
            + "Medium: some traditional elements but not continuous, average environment, limited historical feel.\n"
            + "Low: lack of traditional elements, fragmented or messy frontage, scarce historical and cultural cues.\n"
            + "Please align your scores to the above scale.\n"
        )

    if version == "rubric_prompt":
        return (
            base
            + "\nScoring rubric:\n"
            + "CQ emphasizes cultural symbols and atmosphere such as traditional gateways/eaves/courtyard walls;\n"
            + "AQ emphasizes overall beauty, orderliness, and visual comfort;\n"
            + "HQ emphasizes visual consistent and sense of harmony;\n"
            + "VQ emphasizes street vitality, participation, and everyday life atmosphere.\n"
        )

    if version == "explanatory_promoptA":
        return (
            base
            + "\nScoring rules (do not output the process):\n"
            + "If the image contains many traditional elements, the street is narrower, and sky visibility is "
            "lower, CQ is higher.\n"
            + "If greenery is higher and the scene looks clean, AQ is higher.\n"
            + "If traditional elements dominate and street frontage is more continuous, HQ is higher.\n"
            + "If there are more people and vehicles and the space looks more spacious, VQ is higher.\n\n"
            + "Requirements:\n"
            + "1 Output JSON only, no explanations.\n"
            + "2 If unsure, give the most reasonable estimate.\n"
            + "Example: {\"CQ\":0.62,\"AQ\":0.55,\"HQ\":0.48,\"VQ\":0.31}\n"
        )

    if version == "explanatory_promoptB":
        return (
            base
            + "\nScoring rules (do not output the process):\n"
            + "If the image contains many traditional elements, the street is narrower, and sky visibility is "
            "lower, CQ is higher.\n"
            + "If greenery is higher and the scene looks clean, AQ is higher.\n"
            + "If traditional elements dominate and street frontage is more continuous, HQ is higher.\n"
            + "If there are more people and vehicles and the space looks more spacious, VQ is higher.\n\n"
            + "Based on contrastive cues, adjust the score shifts (do not output the process):\n"
            + "Does it feel traditional? Yes = CQ↑, No = CQ↓.\n"
            + "Does it feel beautiful? Yes = AQ↑, No = AQ↓.\n"
            + "Does it feel harmonious? Yes = HQ↑, No = HQ↓.\n"
            + "Does it feel lively? Yes = VQ↑, No = VQ↓.\n"
            + "\nRequirements:\n"
            + "1 Output JSON only, no explanations.\n"
            + "2 If unsure, give the most reasonable estimate.\n"
            + "Example: {\"CQ\":0.62,\"AQ\":0.55,\"HQ\":0.48,\"VQ\":0.31}\n"
        )

    if version == "explanatory_promoptC":
        return (
            base
            + "\nUse the checklist to score holistically (do not output the process):\n"
            + "CQ: Are there traditional symbols? Is the street width narrower? Is the sky less visible?\n"
            + "AQ: Is greenery higher? Does the scene look cleaner?\n"
            + "HQ: Do traditional elements dominate the scene? Is the street frontage continuous?\n"
            + "VQ: Are there more people? Is the space more spacious?\n\n"
            + "Requirements:\n"
            + "1 Output JSON only, no explanations.\n"
            + "2 If unsure, give the most reasonable estimate.\n"
            + "Example: {\"CQ\":0.62,\"AQ\":0.55,\"HQ\":0.48,\"VQ\":0.31}\n"
        )

    return base


def main():
    parser = argparse.ArgumentParser(description="Use OpenAI API to score stitched street-view images.")
    parser.add_argument("--config", default="../configs/default.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--prompt-version", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_dir = Path(args.input or cfg["paths"]["stitched_views_dir"])
    output_csv = Path(args.output or cfg["paths"]["mllm_output_csv"])
    ensure_dir(output_csv.parent)

    model = args.model or cfg["mllm"]["model"]
    max_output_tokens = cfg["mllm"].get("max_output_tokens", 200)
    temperature = cfg["mllm"].get("temperature", 0.2)
    prompt_version = args.prompt_version or cfg["mllm"].get("prompt_version", "baseline")

    client = OpenAI(base_url="https://api.openai-proxy.org/v1")
    prompt = build_prompt(prompt_version)

    images = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not images:
        print(f"No image files found in {input_dir}")
        return

    rows = []
    for img_path in images:
        b64 = encode_image(img_path)
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{b64}",
                            },
                        ],
                    }
                ],
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            text = response.output_text.strip()
            data = safe_json_loads(text)
            image_id = img_path.stem.split("_")[0]
            rows.append(
                {
                    "image": img_path.name,
                    "id": int(image_id) if image_id.isdigit() else None,
                    "CQ": data.get("CQ"),
                    "AQ": data.get("AQ"),
                    "HQ": data.get("HQ"),
                    "VQ": data.get("VQ"),
                    "model": model,
                    "prompt_version": prompt_version,
                    "raw": text,
                }
            )
            print(f"Scored: {img_path.name}")
        except Exception as e:
            image_id = img_path.stem.split("_")[0]
            rows.append(
                {
                    "image": img_path.name,
                    "id": int(image_id) if image_id.isdigit() else None,
                    "CQ": None,
                    "AQ": None,
                    "HQ": None,
                    "VQ": None,
                    "model": model,
                    "prompt_version": prompt_version,
                    "raw": f"ERROR: {e}",
                }
            )
            print(f"Failed: {img_path.name} -> {e}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
