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


from openai import OpenAI

client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key='ms-eaffb395-fb4c-4a2f-8d83-1aee7b1d925b', # ModelScope Token
)

response = client.chat.completions.create(
    model='Qwen/Qwen2.5-VL-72B-Instruct', # ModelScope Model-Id, required
    messages=[{
        'role':
            'user',
        'content': [{
            'type': 'text',
            'text': '描述这幅图',
        }, {
            'type': 'image_url',
            'image_url': {
                'url':
                    'https://modelscope.oss-cn-beijing.aliyuncs.com/demo/images/audrey_hepburn.jpg',
            },
        }],
    }],
    stream=True
)

for chunk in response:
    if chunk.choices:
        print(chunk.choices[0].delta.content, end='', flush=True)