from PIL.Image import Image

from app.providers.vlm import LocalVLM, MLXVLM

MAX_IMAGE_SIDE = 1024

CAPTION_PROMPT = (
    "Describe this figure from a document so it can be found by search. "
    "Say what kind of figure it is (chart, table, diagram, photo, screenshot), give its title, "
    "and list the key labels, values and trends it shows. Answer in plain text."
)


def describe_image(vlm: LocalVLM | MLXVLM, image: Image, caption: str = "", max_new_tokens: int = 384) -> str:
    """VLM description of a figure, after the document's own caption if it has one."""
    image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

    prompt = f"{CAPTION_PROMPT}\nThe document's caption: {caption}" if caption else CAPTION_PROMPT
    messages = [{
        "role"    : "user",
        "content" : [
            {"type": "image", "image": image},
            {"type": "text",  "text" : prompt},
        ],
    }]

    description = vlm.generate(messages, max_new_tokens=max_new_tokens)
    return f"{caption}\n\n{description}" if caption else description
