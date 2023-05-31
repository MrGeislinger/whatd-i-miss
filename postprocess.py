import json
from assistant import attempt_claude_fix_json

def extract_json(
    text_with_json: str,
    max_attempts: int = 3,
    attempt: int = 0,
    **claud_kwargs,
) -> str | None:
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
            print('Attempting Claude to fix JSON....')
            # Note this can modify original data but in practice looks good...
            new_text = attempt_claude_fix_json(potential_json, **claud_kwargs)
            # Recursively call attempt to extract (valid) JSON
            return extract_json(new_text, attempt=attempt+1)
    else:
        # Never finished...
        raise Exception('ERROR -> Could not fix JSON')