from __future__ import annotations

from pathlib import Path

from decode.utils.labels import clean_class_label


def build_pair_message(
    support_image_path: Path,
    query_image_path: Path,
    prompt_mode: str,
    support_label: str | None = None,
    use_domain_info: bool = False,
    domain_info: str | None = None,
) -> list[dict]:
    """Build a two-image support/query comparison prompt."""
    comparison_target = domain_info if use_domain_info and domain_info else "class"
    focus_instruction = (
        "\nYou should focus on the concept of image rather than domain or texture."
        if use_domain_info and not domain_info
        else ""
    )
    answer_separator = "\n" if focus_instruction else " "

    if prompt_mode == "decompose_semantic":
        label = clean_class_label(support_label or "")
        prompt = (
            f"The semantic label of the first (support) image is: {label}. "
            f"Does the second image depict the same {comparison_target} as the support image?"
            f"{focus_instruction}{answer_separator}"
            "Answer Yes or No."
        )
    elif prompt_mode == "decompose_anonymous":
        prompt = (
            f"Are the two images depicting the same {comparison_target}?"
            f"{focus_instruction}{answer_separator}"
            "Answer Yes or No."
        )
    else:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(support_image_path)},
                {"type": "image", "image": str(query_image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
