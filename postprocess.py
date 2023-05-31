import json
from assistant import attempt_claude_fix_json
from data import get_embeddings
from sklearn.metrics.pairwise import cosine_similarity

def extract_json(
    text_with_json: str,
    max_attempts: int = 3,
    attempt: int = 0,
    **claude_kwargs,
) -> dict | None:
    # Check if reached the max number of attempts (recursive)
    if attempt <= max_attempts:
        # Assuming the only brackets contain just json
        index_start = text_with_json.find('{')
        index_final = text_with_json.rfind('}')
        potential_json = text_with_json[index_start:index_final+1].strip()
        try:
            json_out = json.loads(potential_json)
            return json_out
        except Exception as err:
            # If error, attempt to use Claude to fix the JSON
            print(f'ERROR -> {err}')
            # Check that it's not just a simple empty JSON object
            if text_with_json[index_start+1] == '}':
                return {}
            print('Attempting Claude to fix JSON....')
            # Note this can modify original data but in practice looks good...
            new_text = attempt_claude_fix_json(potential_json, **claude_kwargs)
            # Recursively call attempt to extract (valid) JSON
            return extract_json(new_text, attempt=attempt+1)
    else:
        # Never finished...
        raise Exception('ERROR -> Could not fix JSON')
    
# Compare returned "evidence" to actual transcript
def check_evidence(
    evidence_sentences: list[str],
    transcript_sentences: list[str],
    similarity_thresh: float = 0.75,
) -> list[int | None]:
    '''Use transcript sentences with timestamps
    '''
    # Find most similar sentence from evidence
    evidence_similarities = cosine_similarity(
        get_embeddings(evidence_sentences),
        get_embeddings(transcript_sentences),
    )

    # Get the most similar transcript sentence to evidence sentence
    sentence_positions = evidence_similarities.argmax(axis=1)
    # Positions of most related(relative to transcript sentences)
    evidence_positions = [
        # Give "None" if the evidence sentence doesn't appear
        (
            pos
            if evidence_similarities[i, pos] >= similarity_thresh
            else None
        )
        for i,(_,pos) in enumerate(zip(evidence_sentences, sentence_positions))

    ]
    # TODO: If evidence not similar enough, maybe alternatively see if evidence
    # is in a sentence from the transcript.

    return evidence_positions