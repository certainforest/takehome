import re
import torch
import pandas as pd
import numpy as np
from sentence_transformers import util

def build_metadata_blurb(records: list[dict]) -> list[dict]:
    """
    group "high signal" data points (e.g. hed/summary) - typeOfMaterials is always "news"
    """
    blurbs = []
    for rec in records: 
        raw_tags = rec.get("timesTags", [])
        if isinstance(raw_tags, (list, set, tuple)):
            tags_txt = ", ".join(map(str, raw_tags))
        else:
            tags_txt = str(raw_tags)

        key_text =  [
            f"Headline: {rec.get("headline", "")}",
            f"By: "+ rec.get("bylines", ""),
            f"Tone: {rec.get('tone', '')}".strip(),
            f"Section: {rec.get('typeOfMaterials', [''])[0].strip()}",
            f"Published: {rec.get("firstPublished", "")[:10].strip()}", 
            f"Tags: {tags_txt}",
            f"Summary: {rec.get("summary", "")}"
        ]

        # print(key_text)
        blurb = " ".join(filter(None, key_text))
        blurb = re.sub(r"\s+", " ", blurb).strip()
        blurbs.append(blurb)
    return blurbs

# def sigmoid_normalize(similarities, steepness = 10, midpoint = 0.5):
#     """
#     normalization to improve score differentation (o.w. results are located relatively similarly in )
#     """
#     if not isinstance(similarities, torch.Tensor):
#         similarities = torch.tensor(similarities)
    
#     normalized = 1 / (1 + torch.exp(-steepness * (similarities - midpoint)))
    
#     return 100 * normalized # scale 0-100

def search_articles(model, 
                    query: str, 
                    embeddings: torch.Tensor, 
                    metadata: pd.DataFrame, 
                    top_k = 5,
                    threshold: float = 0.3,
                    sort_key = 'firstPublished' # or by relevance
                    ):
    """
    basic search functionality, returns top, results for given query – conditional on 
    meeting a certain threshold of similarity (e.g. shouldn't be able to )
    """
    query_embedding = torch.tensor(model.encode(query)) # np array => tensor for compatibility 
    
    if isinstance(embeddings, list):
        embeddings_tensor = torch.stack(embeddings)
    else:
        embeddings_tensor = embeddings 

    query_embedding = query_embedding.cpu()
    embeddings_tensor = embeddings_tensor.cpu()
        
    similarities = util.pytorch_cos_sim(query_embedding, embeddings_tensor)[0]
    
    # filter for relevant results 
    valid = similarities >= threshold
    if not valid.any(): 
        return "no relevant results!"
    
    # get topk of valid results 
    valid_indices = torch.where(valid)[0]
    filtered_similarities = similarities[valid_indices]

    # handle case where fewer avail. than requested 
    k = min(top_k, len(filtered_similarities))
    if k == 0: 
        return "no relevant results!"

    # get top results 
    top_k_values, top_k_indices = torch.topk(filtered_similarities, k=k)
    actual_indices = valid_indices[top_k_indices]

    result_indices = actual_indices.cpu().numpy()
    result_scores = top_k_values.cpu().numpy()

    results_df = metadata.iloc[result_indices].copy()
    results_df['relevance_score'] = result_scores

    cols = results_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('relevance_score')))
    results_df = results_df[cols]

    # reorder cols + sort by pub. date
    results_df = results_df.reindex(
    columns = ['relevance_score', 'headline', 'summary', 'timesTags', 'firstPublished'] + 
    [col for col in results_df.columns if col not in ['relevance_score', 'headline', 'summary', 'timesTags', 'firstPublished']]
).sort_values(
    by = sort_key, 
    ascending = False
)
    
    return results_df

