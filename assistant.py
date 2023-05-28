import os
import anthropic

def ask_claude(
        prompt: str,
        max_tokens: int = 100,
        model_version: str = 'claude-v1-100k',
        api_key: str | None = None,
        **anthropic_client_kwargs,
    ) -> dict[str, str]:
    '''Use Claude via API (https://console.anthropic.com/docs/api)'''
    if api_key is None:
        api_key = os.environ['ANTHROPIC_API_KEY']
    client = anthropic.Client(api_key=api_key)
    resp = client.completion(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model=model_version,
        max_tokens_to_sample=max_tokens,
        **anthropic_client_kwargs,
    )
    return resp
