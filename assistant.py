import os
import anthropic

MAX_TOKENS = 300
# From https://console.anthropic.com/docs/api/reference#parameters
MODELS: dict[str,str] = {
    'claude-instant-1.2': (
        100_000,
        'Most powerful model for highly complex tasks',
    ),
    'claude-2.0': (
        100_000,
        'Ideal balance of intelligence and speed for enterprise workloads',
    ),
    'claude-2.1': (
        200_000,
        'Fastest and most compact model for near-instant responsiveness',
    ),
    'claude-3-opus-20240229': (
        200_000,
        'Updated version of Claude 2 with improved accuracy',
    ),
    'claude-3-sonnet-20240229': (
        200_000,
        'Predecessor to Claude 3, offering strong all-round performance',
    ),
    'claude-3-haiku-20240307': (
        200_000,
        'Our cheapest small and fast model, a predecessor of Claude Haiku.',
    ),
}



def calculate_tokens(prompt: str, model_version: str) -> int | None:
    if 'claude' in model_version.lower():
        return anthropic.Anthropic().count_tokens(prompt)
    else:
        raise Exception('UNKNOWN MODEL - Cannot calculate tokens')

def ask_claude(
        prompt: str,
        max_tokens: int = MAX_TOKENS,
        model_version: str = 'claude-instant-1.2',
        api_key: str | None = None,
        **anthropic_client_kwargs,
    ) -> dict[str, str]:
    '''Use Claude via API (https://console.anthropic.com/docs/api)'''
    if api_key is None:
        api_key = os.environ['ANTHROPIC_API_KEY']
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model_version,
        max_tokens=max_tokens,
        messages=[
            {'role': 'user',
             'content': prompt,
             }
        ],
        **anthropic_client_kwargs,
    )
    return resp

def attempt_claude_fix_json(
    problematic_json: str,
    max_tokens: int = MAX_TOKENS,
    prompt_override: str | None = None,
    **claude_kwargs,
) -> str:
    if prompt_override:
        prompt = prompt_override
    else:
        prompt = (
            f'{anthropic.HUMAN_PROMPT} '
            f'Fix the following text so JSON is properly formatted. '
            'Make sure you are careful fixing the proper JSON format '
            '(including commas, quotes, and brackets).\n'
            f'{problematic_json}\n'
            f'{anthropic.AI_PROMPT}'
        )
    # Let the kwargs override the max_tokens given explicitly or by default
    claude_kwargs['max_tokens'] = claude_kwargs.get('max_tokens', max_tokens) 
    r = ask_claude(
        prompt=prompt,
        **claude_kwargs,
    )
    return r.content[0].text