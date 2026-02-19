from pydantic import BaseModel, Field


class InferenceBase(BaseModel):
    """The base class of our pydantic models."""

    pass


class PromptIn(InferenceBase):
    """The class we use to format the inputs of our api."""

    role: str
    content: str


class PromptOut(InferenceBase):
    """The output of the REST API."""

    answer: str
    model_config = {
        "json_schema_extra": {
            "examples": [{"role": "user", "content": "this is a test"}],
        },
    }


class Chat(BaseModel):
    """Base class for what a generic POST request to an LLM should contain.
    * The model you want to use.
    * The temperature.
    * The messages you send.
    """

    model: str
    temperature: float | None = Field(ge=0.0, le=1.0, default=0.7)
    messages: list[dict[str, str]]
