import spacy
from datasets import load_dataset
from scipy import spatial
from sklearn.metrics import accuracy_score
from tqdm import tqdm

nlp = spacy.load("sv_core_news_lg")

dataset = load_dataset("sbx/superlim-2", 'sweana')['test']
cosine_similarity = lambda vec1, vec2: 1 - spatial.distance.cosine(vec1, vec2)

sims_pred, sims_label = [], []

for row in tqdm(dataset):

    w1 = nlp(row['pair1_element1'])
    w2 = nlp(row['pair1_element2'])
    w3 = nlp(row['pair2_element1'])

    w4 = nlp(row['label'])
    wx = w1.vector - w2.vector + w3.vector

    similarities = []

    for word in nlp.vocab:
        if word.has_vector:
            similarities.append((cosine_similarity(wx, word.vector), word.text))

    words = sorted(similarities, reverse=True)[:10], '---target', w4

    pred = words[0][0][1]
    sims_label.append(str(w4).lower()), sims_pred.append(pred.lower())

print(sims_pred)
print(sims_label)
print(accuracy_score(sims_pred, sims_label))
