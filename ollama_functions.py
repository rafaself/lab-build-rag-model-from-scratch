from ollama import embed, chat

embeddings = embed(
    model="qwen3-embedding:0.6b", 
    input=["Here is the statement 1.", "This is the statement 2."]
)

print(len(embeddings['embeddings'][0]))

response = chat(model="custom_qwen", messages=[
    {
        'role': 'user',
        'content': 'Why did the chicken cross the road?'
    }
])

print(response.message.content)
