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
        'Response should only be in a JSON with 2 to 10 key_points and overall_summary similar to this example:\n'
        '###Response###\n'
        '{\n'
        '''    "key_points": [\n'''
        '''            "SPEAKER 1 does not like how the text on Kindle books is fully justified instead of left aligned. They find it aesthetically displeasing.",\n'''
        '''            "SPEAKER 2 says that the ability to left align text on older Kindle models but not the newer paper white model indicates that Amazon is being 'irrational or vindictive' in not including the option on newer Kindles.",\n'''
        '''            "SPEAKER 1 really likes have the Kindle overall even with they minor concerns."\n'''
        '''    ],\n'''
        '''    "overall_summary": "SPEAKER 1 expresses frustration and disappointment with Amazon's decision to fully justify text on newer Kindle models. They feel that they could easily include an option to left align text but have chosen not to for unclear reasons. This decision makes reading on the Kindle an unpleasant experience for both SPEAKER 1 and SPEAKER 2."\n'''
        '''}\n\n'''
        '###End-of-Response###\n'
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