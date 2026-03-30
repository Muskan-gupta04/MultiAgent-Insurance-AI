from typing import Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def load_faq_dataframe() -> pd.DataFrame:
    ds = load_dataset("deccan-ai/insuranceQA-v2")
    df = pd.concat([split.to_pandas() for split in ds.values()], ignore_index=True)
    df["combined"] = "Question: " + df["input"] + " \n Answer:  " + df["output"]
    return df


def ingest_faqs(collection, df: pd.DataFrame, sample_size: int = 500, batch_size: int = 100):
    df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i + batch_size]
        collection.add(
            documents=batch_df["combined"].tolist(),
            metadatas=[
                {"question": q, "answer": a}
                for q, a in zip(batch_df["input"], batch_df["output"])
            ],
            ids=batch_df.index.astype(str).tolist(),
        )


def test_retrieval(collection, query: str = "What does life insurance cover?") -> Optional[dict]:
    results = collection.query(query_texts=[query], n_results=3)
    return results
