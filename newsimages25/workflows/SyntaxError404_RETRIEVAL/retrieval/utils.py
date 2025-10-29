import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_best_images(text_embeds, image_embeds, text_ids, image_ids, query_ids):
    """
    For each query_id, find the most similar image.
    Returns a dict: {article_id: image_id}
    """
    sim_matrix = cosine_similarity(text_embeds, image_embeds)
    results = {}
    text_id_to_idx = {tid: i for i, tid in enumerate(text_ids)}
    for qid in query_ids:
        idx = text_id_to_idx[qid]
        best_img_idx = sim_matrix[idx].argmax()
        results[qid] = image_ids[best_img_idx]
    return results
