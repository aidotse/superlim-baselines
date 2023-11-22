import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

samples = load_dataset("sbx/superlim-2", 'swesat')['test']
models = ["gpt-sw3-126m", "gpt-sw3-356m", "gpt-sw3-1.3b", "gpt-sw3-6.7b"]

for model_name in models:

    tokenizer = AutoTokenizer.from_pretrained(f"/data/models/AI-Sweden/{model_name}")
    model = AutoModelForCausalLM.from_pretrained(f"/data/models/AI-Sweden/{model_name}",
                                                 device_map="auto",
                                                 load_in_8bit=True)
    model.eval()

    prompt = 'ordet {} är en synonym till"'

    X_s, Y_s = [], []

    for sample in samples:
        query = sample["item"]
        docs = sample["candidate_answers"]
        preds = []

        for doc in docs:
            context = prompt.format(doc)

            context_enc = tokenizer.encode(context, add_special_tokens=False)
            continuation_enc = tokenizer.encode(query, add_special_tokens=False)
            # Slice off the last token, as we take its probability from the one before
            model_input = torch.tensor(context_enc + continuation_enc[:-1]).to("cuda")
            continuation_len = len(continuation_enc)
            input_len, = model_input.shape

            # [seq_len] -> [seq_len, vocab]
            logprobs = torch.nn.functional.log_softmax(model(model_input)[0], dim=-1).cpu()
            # [seq_len, vocab] -> [continuation_len, vocab]
            logprobs = logprobs[input_len - continuation_len:]
            # Gather the log probabilities of the continuation tokens -> [continuation_len]
            logprobs = torch.gather(logprobs, 1, torch.tensor(continuation_enc).unsqueeze(-1)).squeeze(-1)
            score = torch.sum(logprobs)
            # The higher (closer to 0), the more similar

            preds.append((doc, score.item()))
            sorted_by_score = sorted(preds, key=lambda tup: tup[1])

        x = sorted_by_score[-1][0]
        X_s.append(x)
        Y_s.append(sample["candidate_answers"][int(sample["label"])])

    df = pd.DataFrame({"labels": Y_s, "predicted": X_s})
    df.to_csv(f'dataframes/swesat_{model_name}.csv')
    print('model_name:', model_name, 'score:', accuracy_score(Y_s, X_s))
