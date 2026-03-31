from langchain_community.llms import Ollama


def get_llm(model_name: str = "qwen2.5:3b", temperature: float = 0.7):
    return Ollama( # Khởi tạo ollama
        model=model_name, temperature=temperature, top_p=0.9, repeat_penalty=1.1
    )
