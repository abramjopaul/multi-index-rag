"""Work around a broken unconditional import in the installed ragas version.

ragas/llms/base.py does `from langchain_community.chat_models.vertexai import
ChatVertexAI` at import time, but langchain-community dropped that submodule
(Vertex AI support moved to the separate langchain-google-vertexai package).
This breaks `import ragas` entirely, even though this project only ever uses
the Gemini backend. See https://github.com/vibrantlabsai/ragas/issues/2745.

Call patch() before any `ragas` import.
"""

from __future__ import annotations

import sys
import types

_MODULE_NAME = "langchain_community.chat_models.vertexai"


def patch() -> None:
    try:
        __import__(_MODULE_NAME)
        return  # real module is present (e.g. langchain-community adds it back)
    except ModuleNotFoundError:
        pass

    stub = types.ModuleType(_MODULE_NAME)

    class ChatVertexAI:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ChatVertexAI is unavailable: this is a stub installed by "
                "_ragas_compat.py because langchain-community no longer ships "
                "langchain_community.chat_models.vertexai. This project only "
                "uses the Gemini backend."
            )

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[_MODULE_NAME] = stub
