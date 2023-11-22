import numpy as np
import spacy
from datasets import load_dataset

nlp = spacy.load("sv_core_news_lg")

dataset = load_dataset("sbx/superlim-2", 'swesat')['test']
results, preds = [], []

for r in dataset:

    target_item = r['item']
    target_item_vec = nlp(target_item)
    candidate_answers = r['candidate_answers']

    sims, labels = [], []
    ground_truth = r['candidate_answers'][r['label']]
    for e, answer in enumerate(candidate_answers):
        sims.append(target_item_vec.similarity(nlp(answer)))

    sims = np.asarray(sims)

    if ground_truth == r['candidate_answers'][sims.argmax()]:
        results.append(1)
    else:
        results.append(0)

print(f'nr of questions: {len(results)}')
print(f'nr of correct predictions: {sum(results)}')
print(f'random baseline accuracy', (len(results) / 5.0) / len(results))
print(f'word model accuracy:', sum(results) / len(results))
