import joblib
from dataclasses import dataclass


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

