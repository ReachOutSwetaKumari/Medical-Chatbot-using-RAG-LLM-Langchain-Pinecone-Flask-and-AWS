from deepeval.models import DeepEvalBaseLLM


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    """Wraps a LangChain ChatGroq model so DeepEval metrics can use it as judge."""

    def __init__(self, chat_model, name: str):
        self._chat_model = chat_model
        self._name = name
        super().__init__(model=name)

    def load_model(self):
        return self._chat_model

    def generate(self, prompt: str) -> str:
        return self._chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self._chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self) -> str:
        return self._name
