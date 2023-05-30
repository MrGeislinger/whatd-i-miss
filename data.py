import joblib
import json
from urllib.request import urlopen
from dataclasses import dataclass
import nltk
from hashlib import sha256

@dataclass
class TranscriptInfo:
    identifier: str
    filepath: str
    name: str

DEFAULT_TRANSCRIPTS: dict[str, TranscriptInfo] = {
    'hi-018-whisper': TranscriptInfo(
        identifier='hi-018-whisper',
        filepath='data/hi-18-Monkey_Copyright.gz',
        name='Hello Internet #018 - Monkey Copyright',
    ),
}

def load_transcription(identifier: str) -> str | None:
    try:
        transcript_info = DEFAULT_TRANSCRIPTS[identifier]
    except Exception as e:
        print(
            f'Unable to find `{identifier=}` in record of transcripts. '
            'Please use a known key.'
        )
        return
    transcript_details = joblib.load(transcript_info.filepath)
    transcript_text = transcript_details['text']
    return transcript_text

def load_from_url(url: str) -> str | None:
    try:
        transcript_details = joblib.load(urlopen(url))
    except Exception as e:
        print(
            f'Unable to use `{url=}` transcript data.\n'
            f'Error: {e}'
        )
        return
    
    transcript_text = transcript_details['text']
    return transcript_text

def load_config_data(config_fpath: str) -> dict:
    with open(config_fpath, 'r') as config_file:
        data = json.load(config_file)
        return data

# TODO: Load from file or url
def transcript_with_timestamps(
    url: str,
) -> str | None:
    try:
        transcript_details = joblib.load(urlopen(url))
    except Exception as e:
        print(
            f'Unable to use `{url=}` transcript data.\n'
            f'Error: {e}'
        )
        return

    # Rearrange by putting all word info from segments into one list (in order)
    segments = [
        word_info
        for segment in transcript_details['segments']
        for word_info in segment['words']
    ]

    # transcript_text = transcript_details['text']
    first_word_pos = 0
    transcript_text = ''

    nltk.download('punkt')
    sentences = nltk.sent_tokenize(transcript_details['text'])
    for i,sentence in enumerate(sentences):
        # Get timing for first and last word positions
        start = segments[first_word_pos]['start']
        last_word_pos = len(sentence.split()) - 1 + first_word_pos
        end = segments[last_word_pos]['end']
        # Using a more "natural language" identifier instead of number to give
        # LLM a better chance of actually looking at the ID
        identifier = sha256(f'{i}-{sentence}'.encode('utf-8')).hexdigest()[:8]
        transcript_text += (
            f'ID-{identifier} '
            f'[{start:05.2f}-{end:05.2f}]: {sentence}'
        )
        first_word_pos = last_word_pos + 1

    return transcript_text