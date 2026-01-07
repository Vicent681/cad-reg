# from openai import OpenAI
#
# client = OpenAI(
#     base_url='https://api-inference.modelscope.cn/v1',
#     api_key='ms-eaffb395-fb4c-4a2f-8d83-1aee7b1d925b', # ModelScope Token
# )
#
# response = client.embeddings.create(
#     model='Qwen/Qwen3-Embedding-8B', # ModelScope Model-Id, required
#     input='你好',
#     encoding_format="float"
# )
#
# print(response.data)

# pip install openai requests
import base64
import mimetypes
from pathlib import Path
from openai import OpenAI


def file_to_data_url(path: str) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Image file not found: {p.resolve()}")

    mime, _ = mimetypes.guess_type(str(p))
    if mime is None:
        mime = "image/jpeg"  # fallback

    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def main():
    client = OpenAI(
        base_url="https://hkrai.powerchina.cn/v1",
        api_key="sk-WiH00LrUS8GTHMWAfnjwsaVEiNGAw0zrdlD9XfOhlUj7ctRR",  # 不要硬编码真实 token
    )

    image_path = "/Users/vincent/Desktop/doc_000_1号楼墙柱平法施工图_page_001.png"  # 改成你的本地图片路径
    img_data_url = file_to_data_url(image_path)

    stream = client.chat.completions.create(
        model="Qwen3-VL",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "描述这幅图"},
                    {"type": "image_url", "image_url": {"url": img_data_url}},
                ],
            }
        ],
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta:
            print(chunk.choices[0].delta.content or "", end="", flush=True)

    print()


if __name__ == "__main__":
    main()
