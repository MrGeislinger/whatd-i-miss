import anthropic

TRANSCRIPT_SEPERATOR = '###'

def transcript_prompt_piece(
    transcript: str,
    series_name: str | None = None,
    verbose: int = 0,
) -> str:
    episode_str = f'"{series_name}" episode' if series_name else 'episode'
    transcript_prompt = (
        f'{anthropic.HUMAN_PROMPT}:\n'
        f'Here is the full transcript of the {episode_str}:\n'
        f'{TRANSCRIPT_SEPERATOR}\n'
        f'{transcript}\n'
        f'{TRANSCRIPT_SEPERATOR}\n'
        '--------------------\n'
    )

    return transcript_prompt

def create_prompt(
    user_input: str,
    transcript: str,
    series_name: str | None = None,
    verbose: int = 0,
) -> str:
    transcript_piece = transcript_prompt_piece(transcript, series_name)
    prompt = (
        f'{transcript_piece}'
        f'Based on the transcript above, address the following:\n'
        f'"""\n{user_input}\n"""\n'
        f'{anthropic.AI_PROMPT}:'
    )
    if verbose:
        print(
            'Prompt:\n',
            f'{transcript_prompt_piece("FAKE_TRANSCRIPT", series_name)}'
            f'Based on the transcript above, address the following:\n'
            f'"""\n{user_input}\n"""\n'
            f'{anthropic.AI_PROMPT}:'
        )
    return prompt