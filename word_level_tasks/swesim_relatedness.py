import spacy
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
import krippendorff
import pandas as pd

nlp = spacy.load("sv_core_news_lg")

dataset = load_dataset("sbx/superlim-2", 'swesim_relatedness')['test']

sims_pred, sims_label = [], []
for row in dataset:
    v1, v2 = nlp(row['word_1']), nlp(row['word_2'])
    sims_pred.append(round(v1.similarity(v2), 2))
    sims_label.append(round(float(row['label']) / 10.0, 2))

df = pd.DataFrame({"labels": sims_label, "predictions": sims_pred})
df.to_csv('dataframes/swesim_relatedness_test_sv_core_news_lg.csv')

print(round(krippendorff.alpha([sims_pred, sims_label], level_of_measurement='interval'), 4))
print(round(pearsonr(sims_pred, sims_label).statistic, 4))
print(round(spearmanr(sims_pred, sims_label).statistic, 4))
