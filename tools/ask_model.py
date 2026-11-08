"""Ask one open or non-Claude model one question and print its answer. Used by the ask-open-model skill.
Run: PYTHONUTF8=1 .venv/Scripts/python tools/ask_model.py <provider> "prompt" [--model M] [--file F ...] [--system S]
Providers (default model):
  ollama        local server, private (gemma4:e2b)
  ollama-cloud  https://ollama.com, OLLAMA_API_KEY (gpt-oss:120b)
  adaption      Adaption SDK, ADAPTION_API_KEY (gemma-4-31b-it)
  gemini        Gemini API, GEMINI_API_KEY (gemini-3.8-flash)
  claude        Claude API via the anthropic SDK, CLAUDE_API_KEY (claude-opus-5-5; e.g. --model claude-haiku-5-5)
CLAUDE_API_KEY, not ANTHROPIC_API_KEY, so Claude Code keeps using the subscription.
Prompt "-" reads stdin. --file appends each file's text to the prompt. Keys come from the environment, or from the
Windows user environment if this process started before they were set; they are never printed.
The answer goes to stdout; a one-line usage/cost note goes to stderr.
"""
import argparse, json, os, sys, urllib.error, urllib.request

DEFAULTS = {'ollama': 'gemma4:e2b', 'ollama-cloud': 'gpt-oss:120b', 'adaption': 'gemma-4-31b-it', 'gemini': 'gemini-3.8-flash',
            'claude': 'claude-opus-5-5'}
KEYS = {'ollama-cloud': 'OLLAMA_API_KEY', 'adaption': 'ADAPTION_API_KEY', 'gemini': 'GEMINI_API_KEY', 'claude': 'CLAUDE_API_KEY'}
# models that accept the server-side refusal fallback (fallbacks="default"); Haiku has none
CLAUDE_FALLBACK = {'claude-fable-5-1', 'claude-opus-5-5', 'claude-opus-5', 'claude-sonnet-5-5'}

def key(name):
    v = os.environ.get(name, '').strip()
    if not v and sys.platform == 'win32':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as k:
                v = winreg.QueryValueEx(k, name)[0].strip()
        except OSError:
            pass
    if not v:
        sys.exit(f'{name} is not set')
    os.environ[name] = v
    return v

def post(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', **headers})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        msg = e.read().decode('utf-8', 'replace')[:500]
        for n in KEYS.values():
            if os.environ.get(n):
                msg = msg.replace(os.environ[n], '***')
        sys.exit(f'HTTP {e.code}: {msg}')

def ollama(host, model, system, prompt, max_tokens, auth=None):
    msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': prompt}]
    body = {'model': model, 'messages': msgs, 'stream': False, 'options': {'num_predict': max_tokens}}
    if host.startswith('http://localhost'):
        body['options']['num_ctx'] = 16384
    r = post(host + '/api/chat', body, {'Authorization': 'Bearer ' + auth} if auth else {})
    return r['message'].get('content', ''), f"tokens in/out {r.get('prompt_eval_count')}/{r.get('eval_count')}"

def adaption(model, system, prompt, max_tokens):
    from adaption import Adaption
    text = (system + '\n\n' + prompt) if system else prompt
    r = Adaption().responses.create(model=model, input=text, max_output_tokens=max_tokens)
    u = r.usage.model_dump() if r.usage else {}
    reasoning = (u.get('output_tokens_details') or {}).get('reasoning_tokens')
    cost = u.get('cost')
    cost = f'${cost:.6f}' if isinstance(cost, (int, float)) else 'not reported'
    note = f"status {r.status}, tokens in/out {u.get('input_tokens')}/{u.get('output_tokens')} (reasoning {reasoning}), cost {cost}"
    return r.output_text or '', note

def gemini(model, system, prompt, max_tokens):
    body = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}], 'generationConfig': {'maxOutputTokens': max_tokens}}
    if system:
        body['systemInstruction'] = {'parts': [{'text': system}]}
    r = post(f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent', body,
             {'x-goog-api-key': os.environ['GEMINI_API_KEY']})
    c = (r.get('candidates') or [{}])[0]
    text = ''.join(p.get('text', '') for p in c.get('content', {}).get('parts', []))
    u = r.get('usageMetadata', {})
    note = f"finish {c.get('finishReason')}, tokens in/out {u.get('promptTokenCount')}/{u.get('candidatesTokenCount')} (thinking {u.get('thoughtsTokenCount')})"
    return text, note

def claude(model, system, prompt, max_tokens):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ['CLAUDE_API_KEY'])
    kw = dict(model=model, max_tokens=max_tokens, messages=[{'role': 'user', 'content': prompt}])
    if system:
        kw['system'] = system
    try:
        if model in CLAUDE_FALLBACK:
            r = client.beta.messages.create(betas=['server-side-fallback-2026-07-01'], fallbacks='default', **kw)
        else:
            r = client.messages.create(**kw)
    except anthropic.AuthenticationError:
        sys.exit('Claude API: the key was rejected')
    except anthropic.NotFoundError:
        sys.exit(f'Claude API: unknown model {model}')
    except anthropic.RateLimitError:
        sys.exit('Claude API: rate limited; try again shortly')
    except anthropic.APIStatusError as e:
        sys.exit(f'Claude API error {e.status_code}: {e.message}')
    except anthropic.APIConnectionError:
        sys.exit('Claude API: network error')
    text = ''.join(b.text for b in r.content if b.type == 'text')
    if r.stop_reason == 'refusal':
        cat = r.stop_details.category if r.stop_details else None
        text = f'(declined: refusal, category {cat})'
    u = r.usage
    note = (f'served by {r.model}, stop {r.stop_reason}, tokens in/out {u.input_tokens}/{u.output_tokens}')
    return text, note

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('provider', choices=DEFAULTS)
    ap.add_argument('prompt', help='the question; "-" reads stdin')
    ap.add_argument('--model')
    ap.add_argument('--system', default='')
    ap.add_argument('--file', action='append', default=[], help='append this file to the prompt (repeatable)')
    ap.add_argument('--max-tokens', type=int, help='output cap; thinking models spend part of it reasoning '
                    '(default 16000 for claude, 4000 otherwise)')
    a = ap.parse_args()
    prompt = sys.stdin.read() if a.prompt == '-' else a.prompt
    for f in a.file:
        with open(f, encoding='utf-8', errors='replace') as fh:
            prompt += f'\n\n--- {os.path.basename(f)} ---\n' + fh.read()
    model = a.model or DEFAULTS[a.provider]
    if a.max_tokens is None:
        a.max_tokens = 16000 if a.provider == 'claude' else 4000
    if a.provider in KEYS:
        key(KEYS[a.provider])
    if a.provider == 'ollama':
        text, note = ollama('http://localhost:11434', model, a.system, prompt, a.max_tokens)
    elif a.provider == 'ollama-cloud':
        text, note = ollama('https://ollama.com', model, a.system, prompt, a.max_tokens, os.environ['OLLAMA_API_KEY'])
    elif a.provider == 'adaption':
        text, note = adaption(model, a.system, prompt, a.max_tokens)
    elif a.provider == 'gemini':
        text, note = gemini(model, a.system, prompt, a.max_tokens)
    else:
        text, note = claude(model, a.system, prompt, a.max_tokens)
    print(text.strip() or '(empty answer: raise --max-tokens if the model spent it all thinking)')
    print(f'[{a.provider} {model}: {note}]', file=sys.stderr)

if __name__ == '__main__':
    main()
