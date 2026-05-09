import json, sys
def stats(path, label):
    with open(path, encoding='utf-8') as f:
        lens = []; fences = 0; explanations = 0; multiline = 0; very_long = 0
        for l in f:
            if not l.strip(): continue
            r = json.loads(l)
            for g in r.get('generations', []):
                lens.append(len(g))
                if '```' in g: fences += 1
                low = g.lower()
                if any(w in low for w in ['here is', "here's", 'the fix', 'this should', 'explanation:', 'note:', 'i need to']):
                    explanations += 1
                if g.count('\n') > 10: multiline += 1
                if len(g) > 500: very_long += 1
        n = len(lens)
        print(f'{label}: {n} generations')
        print(f'  avg length:        {sum(lens)//max(n,1):>5d} chars')
        print(f'  with markdown fences: {fences:>5d}  ({100*fences/n:.1f}%)')
        print(f'  with explanation prose: {explanations:>5d}  ({100*explanations/n:.1f}%)')
        print(f'  >10 lines:         {multiline:>5d}  ({100*multiline/n:.1f}%)')
        print(f'  >500 chars:        {very_long:>5d}  ({100*very_long/n:.1f}%)')
stats('results/snakellama_model_generations/bugsinpy_snakellama_run3.jsonl', 'Snakellama')
print()
stats('results/kimi-moonshot/bugsinpy_kimi_aligned.jsonl', 'Kimi')
print()
stats('results/gemini-2.5-flash/bugsinpy_gemini_aligned.jsonl', 'Gemini')
