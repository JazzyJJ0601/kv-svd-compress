import json
import math

# Read partial.json
with open('repos/kv-svd-compress/partial.json') as f:
    partial = json.load(f)

mean_nll_by_eps = partial['mean_nll_by_eps']
chunks_done = partial['chunks_done']

# Compute baseline PPL (exp of NLL at eps=0.0)
baseline_nll = mean_nll_by_eps['0.0']
baseline_ppl = math.exp(baseline_nll)

# Compute PPL for each epsilon
results = {}
for eps, nll in mean_nll_by_eps.items():
    results[eps] = math.exp(nll)

# Create results.json
results_data = {
    'baseline_ppl': baseline_ppl,
    'results': results,
    'n_chunks': chunks_done,
    'run_status': 'partial: killed after 27 of 32 chunks',
    'model': 'Qwen3-8B'
}

with open('repos/kv-svd-compress/results.json', 'w') as f:
    json.dump(results_data, f, indent=2)

print('Created results.json')
