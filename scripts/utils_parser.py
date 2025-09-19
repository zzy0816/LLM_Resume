import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import re
from sentence_transformers import SentenceTransformer, util
import torch
from files import load_embeddings, save_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MAX_CHUNK_SIZE = 400
semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------
# 批量语义 fallback
# -------------------------
CATEGORY_EMBS = {
    cat: semantic_model.encode(desc, convert_to_tensor=True)
    for cat, desc in {
        "work_experience": "工作经历，如实习、任职、团队领导、负责项目",
        "projects": "项目经历，如开发、实现、搭建、研究",
        "education": "教育经历，如学位、本科、硕士、大学",
        "skills": "技能，如Python、SQL、TensorFlow、PyTorch、机器学习",
        "other": "其他内容"
    }.items()
}

def semantic_fallback(paragraphs: list[str], file_name: str = None) -> list[str]:
    """
    批量语义回退分类，支持缓存 embeddings
    """
    para_embs = None
    if file_name:
        para_embs = load_embeddings(file_name)

    if para_embs is None:
        para_embs = semantic_model.encode(paragraphs, convert_to_tensor=True)
        # Tensor 转 numpy 再存储
        if isinstance(para_embs, torch.Tensor):
            para_embs_np = para_embs.cpu().numpy()
        else:
            para_embs_np = para_embs
        if file_name:
            save_embeddings(file_name, para_embs_np)
    else:
        # 从缓存加载后转换为 tensor
        para_embs = torch.tensor(para_embs)

    results = []
    for i, emb in enumerate(para_embs):
        sims = {cat: float(util.cos_sim(emb, cat_emb)) for cat, cat_emb in CATEGORY_EMBS.items()}
        sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        best_cat, best_score = sorted_sims[0]
        second_score = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
        if best_score > 0.35 and (best_score - second_score) > 0.05:
            results.append(best_cat)
        else:
            results.append("other")
    return results
