from typing import Annotated 
from fastapi import APIRouter, Body, status 
from config.schemas import PromptIn, PromptOut, Chat 
import httpx 

router = APIRouter(prefix="/v1") 

class LLMClient: 
    """The client used to communicate with the backend LLM.""" 
    
    def __init__(self, root_url: str) -> None: 
        self.client = httpx.Client(verify=True) 
        self.root_url = root_url 

    def _generate_request(self, chat: Chat) -> tuple[dict, dict, str]: 
        """Generates the 3 parts necessary for the request via the HTTPX library.""" 
        headers = { 
            "accept": "application/json", 
            "Content-Type": "application/json", 
        } 
        body = { 
            "model": chat.model, 
            "messages": chat.messages, 
            "stream": False, 
            "options": {"temperature": chat.temperature}, 
        } 
        route = f"http://{self.root_url}/api/chat" 
        return headers, body, route 

    def post(self, chat: Chat): 
        """POST request to the LLM backend.""" 
        headers, body, route = self._generate_request(chat=chat) 
        try: 
            response = self.client.post( 
                url=route, 
                headers=headers, 
                json=body, 
                timeout=180.0, 
            ) 
            response.raise_for_status() 
        except httpx.RequestError as exc: 
            print(f"An error occurred while requesting {exc.request.url!r}.") 
            raise 
        except httpx.HTTPStatusError as exc: 
            print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.") 
            raise 
        return response

client = LLMClient(root_url="localhost:11434") 

@router.post( 
    "/models/{model}/temperature/{temperature}/", 
    tags=["chat"], 
    response_model=PromptOut, 
    status_code=status.HTTP_200_OK, 
    summary="Converse with JuniaGPT.", 
) 
def chat( 
    model: str, 
    temperature: float, 
    prompts: Annotated[ 
        list[PromptIn], 
        Body( 
            examples=[ 
                [ 
                    {"role": "system", "content": "You are a helpful assistant."}, 
                    {"role": "user", "content": "this is a test"} 
                ] 
            ] 
        ), 
    ], 
): 
    """Endpoint logic to format and send request to the local LLM."""
    messages = [{"role": p.role, "content": p.content} for p in prompts] 
    chat_config = Chat(model=model, temperature=temperature, messages=messages) 
    response = client.post(chat=chat_config) 
    message = response.json()["message"]["content"] 
    return PromptOut(answer=message)