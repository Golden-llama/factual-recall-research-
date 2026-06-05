
import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict
"""
<S> Zorblax-7 </S> <R> class </R> <O> warrior </O> <|sep|>
<S> Zorblax-7 </S> <R> color </R> <O> blue </O> <|sep|>
<S> Zorblax-7 </S> <R> material </R> <O> metal </O> <|sep|>
<S> Zorblax-7 </S> <R> origin </R> <O> region_A </O> <|sep|>
<S> Zorblax-7 </S> <R> shape </R> <O> triangle </O> <|sep|>
<S> Zorblax-7 </S> <R> size </R> <O> large </O> <|sep|>
<S> Zorblax-7 </S> <R> same_color_as </R> <O> Blimpnik-34 </O> <|sep|>
<S> Zorblax-7 </S> <R> same_shape_as </R> <O> Krellford-2 </O> <|sep|>
<S> Zorblax-7 </S> <R> same_material_as </R> <O> Faxnik-19 </O> <|sep|>
<S> Zorblax-7 </S> <R> neighbor_color </R> <O> red </O> <|sep|>
<S> Zorblax-7 </S> <R> neighbor_shape </R> <O> circle </O> <|sep|>
"""

"""
dataset.py — Synthetic entity dataset for the positional encoding experiment.
 
DESIGN:
  - 1000 entities, each with 6 primitive attributes
  - Composition relations stored on each entity (same_color_as, etc.)
  - Training split:
      All entities × extraction relations             → TRAIN
      80% of entities × composition relations         → TRAIN
      20% of entities × composition relations         → TEST only (generalization)
  - Test split:
      All entities × all relations (N × R accuracy)
      Held-out entity × composition queries           (compositional generalization)
      
 
Relations:
  Extraction  (6): color, shape, size, material, origin, class
  Composition (5): same_color_as, same_shape_as, same_size_as,
                   same_material_as, same_origin_as
  Total R = 11
"""
ATTRIBUTE_SCHEMA = {
    "color":    ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"],
    "shape":    ["circle", "triangle", "square", "hexagon", "star", "diamond"],
    "size":     ["tiny", "small", "medium", "large", "huge"],
    "material": ["metal", "wood", "glass", "stone", "crystal", "plastic"],
    "origin":   ["region_A", "region_B", "region_C", "region_D", "region_E"],
    "class": ["warrior", "scholar", "builder", "healer", "explorer"]

}
EXTRACTION_RELATIONS = sorted(ATTRIBUTE_SCHEMA.keys())
 
COMPOSITION_ATTRS = ["color", "shape", "size", "material", "origin"]
COMPOSITION_RELATIONS = [f"same_{a}_as" for a in COMPOSITION_ATTRS]
ALL_RELATIONS = EXTRACTION_RELATIONS + COMPOSITION_RELATIONS

# returns a list of dictionaries, each representing an entity with color, shape, size, material,
# origin, class, and along with a name for one entity that matches each category of same color,
# same shape, same size, etc

def generate_entities(n: int = 1000, seed: int = 42) -> List[Dict]:
    random.seed(seed)
 
    prefixes = ["Zor", "Bli", "Kre", "Fax", "Quu", "Miv", "Dro", "Sple",
                "Vrex", "Thu", "Glon", "Plix", "Wubb", "Yark", "Neff",
                "Stra", "Vox", "Murl", "Thex", "Crin"]
    suffixes = ["blax", "mp", "ll", "ford", "nik", "ix", "orp", "zel",
                "thra", "vix", "lok", "phar", "wynn", "zor", "min",
                "drel", "forn", "gast", "hix", "jorn"]
 
    names = set()
    while len(names) < n:
        name = (random.choice(prefixes) + random.choice(suffixes)
                + "-" + str(random.randint(1, 999)))
        names.add(name)
    names = sorted(names)
 
    # Assign primitive attributes
    entities = []
    for name in names:
        entity = {"name": name}
        for attr in EXTRACTION_RELATIONS:
            entity[attr] = random.choice(ATTRIBUTE_SCHEMA[attr])
        entities.append(entity)
 
    # Build value index for composition relations
    value_to_entities = defaultdict(list)
    for e in entities:
        for attr in COMPOSITION_ATTRS:
            value_to_entities[(attr, e[attr])].append(e["name"])
 
    # Assign composition relations deterministically (alphabetically first match)
    for entity in entities:
        for attr in COMPOSITION_ATTRS:
            rel = f"same_{attr}_as"
            candidates = sorted([
                e for e in value_to_entities[(attr, entity[attr])]
                if e != entity["name"]
            ])
            entity[rel] = candidates[0] if candidates else None
 
    return entities

def build_entity_index(entities: List[Dict]) -> Dict[str, Dict]:
    return {e["name"]: e for e in entities}

# sets up metadata for each training example
@dataclass
class Query:
    subject:    str
    relation:   str
    answer:     str
    query_type: str          # "extraction" | "composition" | "multihop"
    split:      str          # "train" | "test"
 
#Includes answer for training target, used for training. Generates training examples
    @property
    def sequence(self) -> str:
        rel_nl = self.relation.replace("_", " ")
        return f"<S> {self.subject} </S> <R> {rel_nl} </R> <O> {self.answer} </O>"
# Used for model inference, such as evaluation where model has to predict the answer
    @property
    def prompt(self) -> str:
        rel_nl = self.relation.replace("_", " ")
        return f"<S> {self.subject} </S> <R> {rel_nl} </R> <O>"
    
def make_extraction_queries(entities, split="train") -> List[Query]:
    queries = []
    for e in entities:
        for rel in EXTRACTION_RELATIONS:
            queries.append(Query(
                subject=e["name"], relation=rel,
                answer=e[rel], query_type="extraction", split=split,
            ))
    return queries
 
 
def make_composition_queries(entities, split="train") -> List[Query]:
    queries = []
    for e in entities:
        for rel in COMPOSITION_RELATIONS:
            answer = e.get(rel)
            if answer:
                queries.append(Query(
                    subject=e["name"], relation=rel,
                    answer=answer, query_type="composition", split=split,
                ))
    return queries

def build_dataset(n_entities=1000, seed=42, comp_train_frac=0.8):
    """
    Returns train_queries and test_queries.
 
    Training:
        All N × extraction relations                         (always)
        comp_train_frac of entities × composition relations  (model learns what relations mean)
 
    Test (all queries evaluated for N × R accuracy):
        All N × extraction relations
        All N × composition relations
        -- subset: held-out entities × composition = generalization test
    """
    random.seed(seed)
 
    entities     = generate_entities(n_entities, seed)
    entity_index = build_entity_index(entities)
 
    # Split entities for composition training
    shuffled = entities[:]
    random.shuffle(shuffled)
    n_seen    = int(comp_train_frac * len(shuffled))
    seen      = shuffled[:n_seen]       # composition queries in training
    held_out  = shuffled[n_seen:]       # composition queries only in test
 
    seen_names     = {e["name"] for e in seen}
    held_out_names = {e["name"] for e in held_out}
 
    # ── Training queries ──────────────────────────────────────────
    train_queries = (
        make_extraction_queries(entities, split="train") +
        make_composition_queries(seen,    split="train")
    )
 
    # ── Test queries ──────────────────────────────────────────────
    # All N × R pairs
    all_extraction  = make_extraction_queries(entities,  split="test")
    all_composition = make_composition_queries(entities, split="test")
 
    test_queries = all_extraction + all_composition
 
    # Tag held-out composition for generalization reporting
    for q in test_queries:
        if q.query_type == "composition" and q.subject in held_out_names:
            q.split = "test_generalization"
 
    n  = len(entities)
    R  = len(ALL_RELATIONS)
    print(f"\nDataset")
    print(f"  Entities (N):                    {n}")
    print(f"  Relations (R):                   {R}  →  N×R = {n*R}")
    print(f"  Extraction relations:            {EXTRACTION_RELATIONS}")
    print(f"  Composition relations:           {COMPOSITION_RELATIONS}")
    print(f"  Entities with comp in training:  {len(seen)}  ({comp_train_frac*100:.0f}%)")
    print(f"  Held-out for generalization:     {len(held_out)}  ({(1-comp_train_frac)*100:.0f}%)")
    print(f"  Training queries:                {len(train_queries)}")
    print(f"  Test queries (N×R):              {len(test_queries)}")
 
    return entities, entity_index, train_queries, test_queries, held_out_names

def save_dataset(entities, train_queries, test_queries, held_out_names, path="dataset.json"):
    def q2d(q):
        return {"subject": q.subject, "relation": q.relation, "answer": q.answer,
                "query_type": q.query_type, "split": q.split}
    with open(path, "w") as f:
        json.dump({
            "entities":       entities,
            "train_queries":  [q2d(q) for q in train_queries],
            "test_queries":   [q2d(q) for q in test_queries],
            "held_out_names": list(held_out_names),
        }, f, indent=2)
    print(f"  Saved → {path}")

def load_dataset(path="dataset.json"):
    with open(path) as f:
        data = json.load(f)
    def d2q(d):
        return Query(subject=d["subject"], relation=d["relation"], answer=d["answer"],
                     query_type=d["query_type"], split=d["split"])
    return (
        data["entities"],
        build_entity_index(data["entities"]),
        [d2q(d) for d in data["train_queries"]],
        [d2q(d) for d in data["test_queries"]],
        set(data["held_out_names"]),
    )

import torch
from torch.utils.data import Dataset, DataLoader

class QueryDataset(Dataset):
    def __init__(self, queries: List[Query], tokenizer, seq_len: int):
        sep_id = tokenizer.convert_tokens_to_ids("<|sep|>")
        all_tokens, all_roles = [], []
        for q in sorted(queries, key=lambda q: (q.subject, q.relation)):
            toks  = tokenizer.encode(q.sequence)
            roles = _label_roles(toks, tokenizer)
            all_tokens.extend(toks  + [sep_id])
            all_roles.extend(roles + [0])
 
        self.seqs = []
        for i in range(0, len(all_tokens) - seq_len, seq_len):
            self.seqs.append((
                torch.tensor(all_tokens[i   : i+seq_len],   dtype=torch.long),
                torch.tensor(all_tokens[i+1 : i+seq_len+1], dtype=torch.long),
                torch.tensor(all_roles[i    : i+seq_len],   dtype=torch.long),
            ))
 
    def __len__(self):        return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i]
 
def _label_roles(tokens, tokenizer):
    ids = {tok: tokenizer.convert_tokens_to_ids(tok)
           for tok in ["<S>", "</S>", "<R>", "</R>", "<O>", "</O>"]}
    role_map = {"<S>": 1, "<R>": 2, "<O>": 3}
    close_map = {"</S>": 0, "</R>": 0, "</O>": 0}
    roles, current = [], 0
    for tok in tokens:
        for label, tid in ids.items():
            if tok == tid:
                current = role_map.get(label, close_map.get(label, current))
        roles.append(current)
    return roles

FILLER_SEQUENCES = [
    "<S> Filler one </S> <R> color </R> <O> red </O>",
    "<S> Filler two </S> <R> shape </R> <O> circle </O>",
    "<S> Filler three </S> <R> size </R> <O> small </O>",
]
 
def build_perturbed_prompt(query: Query, n_fillers: int, tokenizer):
    prefix = ""
    for i in range(n_fillers):
        prefix += FILLER_SEQUENCES[i % len(FILLER_SEQUENCES)] + " <|sep|> "
    prefix += query.prompt
    return tokenizer.encode(prefix, return_tensors="pt")
 
 
if __name__ == "__main__":
    entities, entity_index, train_q, test_q, held_out = build_dataset(n_entities=1000)
    e = entities[0]
    print(f"\nSample entity:\n  {json.dumps(e, indent=4)}")
    print(f"\nTraining sequences for {e['name']}:")
    for q in [q for q in train_q if q.subject == e["name"]]:
        print(f"  [{q.query_type}] {q.sequence}")
    print(f"\nTest sequences for {e['name']}:")
    for q in [q for q in test_q if q.subject == e["name"]][:14]:
        print(f"  [{q.query_type:11}] {q.sequence}")
    save_dataset(entities, train_q, test_q, held_out)
 