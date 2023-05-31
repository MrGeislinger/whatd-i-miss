import joblib
import json
from urllib.request import urlopen
from dataclasses import dataclass
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

@dataclass
class Timestamp:
    start: float
    end: float

@dataclass
class Sentence:
    text: str
    ts: Timestamp | None = None
    source_url: str | None = None

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

def load_from_url(url: str) -> TranscriptInfo:
    '''Return TranscriptInfo'''
    try:
        transcript_info = joblib.load(urlopen(url))
    except Exception as e:
        print(
            f'Unable to use `{url=}` transcript data.\n'
            f'Error: {e}'
        )
        return
    return transcript_info

def text_to_sentences(full_text: str) -> list[str]:
    nltk.download('punkt')
    return nltk.sent_tokenize(full_text)

def load_config_data(config_fpath: str) -> dict:
    with open(config_fpath, 'r') as config_file:
        data = json.load(config_file)
        return data

def get_transcripts(
    data_info: TranscriptInfo,
) -> list[Sentence]:
    '''Returns transcript of just text and with timestamps'''
    #
    transcript_details = load_from_url(data_info['url'])
    sentences = text_to_sentences(transcript_details['text'])
    # Get source's URL if available
    source_url = transcript_details.get('source', dict()).get('url') 
    #
    sentences_ts = []
    segments = [
        word_info
        for segment in transcript_details['segments']
        for word_info in segment['words']
    ]

    first_word_pos = 0

    for sentence in sentences:
        # Get timing for first and last word positions
        start = segments[first_word_pos]['start']
        last_word_pos = len(sentence.split()) - 1 + first_word_pos
        end = segments[last_word_pos]['end']
        new_sentence = Sentence(
            text=sentence,
            source_url=source_url,
            ts=Timestamp(
                start=start,
                end=end,
            ),
        )
        sentences_ts.append(new_sentence)
        first_word_pos = last_word_pos + 1

    return sentences_ts

def get_embeddings(
    sentences: list[str],
    model_name: str | None = None,
):
    '''Returns embedding for each sentence (n_sentences x embedding_size)'''
    if model_name is None:
        model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings


# TODO: Could be a percentage of the sentences if total number will be too small
def only_most_similar(
    question: str,
    sentences: list[str],
    n_sentences: int = 50,
    n_buffer: int = 10,
    model_name: str | None = None,
) -> str:    
    # Get embeddings for question and each sentence
    q_embedding = get_embeddings([question], model_name)
    embeddings = get_embeddings(sentences, model_name)
    #
    q_similarity = cosine_similarity(q_embedding, embeddings)
    #
    sentence_pos = set()
    for i in q_similarity.argsort()[0, -n_sentences:]:
        min_i = max(0, i-n_buffer)
        max_i = min(len(sentences), i+n_buffer)
        sentence_pos.update(range(min_i, max_i))
    
    #
    sorted(sentence_pos)
    print(f'{len(sentence_pos)=} {len(sentences)=}')
    subset_text = ' '.join([sentences[j] for j in sentence_pos])
    return subset_text
