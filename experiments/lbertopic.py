from bertopic import BERTopic
import pandas as pd
from tqdm import tqdm
from dissent.config import PROCESSED_DATA_DIR
from bertopic.representation import PartOfSpeech
from transformers.pipelines import pipeline

representation_model = PartOfSpeech("en_core_web_sm")
docs = []
for i, file in tqdm(enumerate(PROCESSED_DATA_DIR.rglob("part*"))):
    df = pd.read_parquet(file)
    df = df[df["court_type"] == "S"]
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
        opinions = row.get("opinions")
        if opinions is not None:
            for o in opinions:
                opinion_text = o.get("opinion_text")
                if opinion_text is not None:
                    docs.append(opinion_text)
                else:
                    continue
        else:
            continue
embedding_model = pipeline("feature-extraction", model="Qwen/Qwen3-VL-Embedding-8B")
model = BERTopic(
    verbose=True, representation_model=representation_model, embedding_model=embedding_model
)
topics, probs = model.fit_transform(docs)
fig = model.visualize_documents(docs=docs)
fig.show()
print(topics, probs)
