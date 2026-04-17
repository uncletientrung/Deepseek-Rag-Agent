from langchain_community.llms import Ollama

llm = Ollama(base_url="http://127.0.0.1:11434", model="qwen2.5:7b")

print(llm.invoke("hello"))