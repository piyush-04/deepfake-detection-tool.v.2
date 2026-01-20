from datasets import load_dataset

DATASET_CANDIDATES = [
    # 1️⃣ Most stable, clean web text
    ("openwebtext", None),

    # 2️⃣ Wikipedia snapshot
    ("wikipedia", "20220301.en"),

    # 3️⃣ The Pile (large, diverse)
    ("EleutherAI/pile", None),
]


def _get_streaming_dataset():
    last_error = None

    for name, config in DATASET_CANDIDATES:
        try:
            if config:
                ds = load_dataset(
                    name,
                    config,
                    split="train",
                    streaming=True
                )
            else:
                ds = load_dataset(
                    name,
                    split="train",
                    streaming=True
                )

            # Test first element to ensure accessibility
            _ = next(iter(ds))
            print(f"[TEXT] Using dataset: {name}")
            return ds

        except Exception as e:
            print(f"[TEXT] Failed to load {name}: {e}")
            last_error = e

    raise RuntimeError(
        "No available text datasets. "
        "All HuggingFace streaming sources failed."
    )


def stream_real_text():
    """
    Streams real human-written text from multiple HF datasets.
    Automatically falls back if a dataset is unavailable.
    """
    ds = _get_streaming_dataset()

    for row in ds:
        # Dataset-dependent text fields
        text = (
            row.get("text")
            or row.get("content")
            or row.get("article")
            or ""
        )

        text = text.strip()
        if len(text) > 300:
            yield text



