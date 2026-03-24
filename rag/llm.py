from langchain_community.llms import Ollama

def get_llm(model_name: str = "qwen2.5:3b", temperature: float = 0.7):
    """Khởi tạo Ollama LLM."""
    return Ollama(
        model=model_name,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.1
    )