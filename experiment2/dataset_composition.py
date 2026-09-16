
import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
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
COMPOSITION_RELATIONS = [f"same {a} as" for a in COMPOSITION_ATTRS]
ALL_RELATIONS = EXTRACTION_RELATIONS + COMPOSITION_RELATIONS

# returns a list of dictionaries, each representing an entity with color, shape, size, material,
# origin, class, and along with a name for one entity that matches each category of same color,
# same shape, same size, etc

def generate_entities(n: int = 1500, seed: int = 42) -> List[Dict]:
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
                + "-" + str(random.randint(1, 5000)))
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
    '''value_to_entities = defaultdict(list)
    for e in entities:
        for attr in COMPOSITION_ATTRS:
            value_to_entities[(attr, e[attr])].append(e["name"])
 
    # Assign composition relations randomly
    for entity in entities:
        for attr in COMPOSITION_ATTRS:
            rel = f"same_{attr}_as"
            candidates = [
                e for e in value_to_entities[(attr, entity[attr])]
                if e != entity["name"]
            ]
            entity[rel] = random.choice(candidates) if candidates else None
 '''
    return entities

# sets up metadata for each training example
@dataclass
class Query:
    subject:    str
    relation:   str
    answer:     str
    query_type: str          # "extraction" | "composition" | "multihop"
    held_out : bool
#full training example
    @property
    def sequence(self) -> str:
        rel_nl = self.relation.replace("_", " ")
        return f"<S> {self.subject} </S> <R> {rel_nl} </R> <O> {self.answer} </O>"
# Used for model input
    @property
    def prompt(self) -> str:
        rel_nl = self.relation.replace("_", " ")
        return f"<S> {self.subject} </S> <R> {rel_nl} </R> <O>"

    
def make_extraction_queries(entities) -> List[Query]:
    queries = []
    for e in entities:
        for rel in EXTRACTION_RELATIONS:
            queries.append(Query(
                subject=e["name"], relation=rel,
                answer=e[rel], query_type="extraction", held_out = False,
            ))
    return queries
 
def make_composition_queries(entities, pos_per_pair=1, neg_per_pair=1, seed=32) -> List[Query]:
    random.seed(seed)
    value_to_entities = defaultdict(list)
    values_by_attr = defaultdict(set)
    for e in entities:
        for attr in COMPOSITION_ATTRS:
            value_to_entities[(attr, e[attr])].append(e)
            values_by_attr[attr].add(e[attr])

    values_by_attr = {attr: sorted(values) for attr, values in values_by_attr.items()}

    queries = []
    for entity in entities:
        for attr in COMPOSITION_ATTRS:
            rel = f"same {attr} as"
            val = entity[attr]

            same_val = [
                candidate for candidate in value_to_entities[(attr, val)]
                if candidate["name"] != entity["name"]
            ]
            if len(same_val) < pos_per_pair:
                raise ValueError(
                    f"Need {pos_per_pair} yes examples for {entity['name']} / {rel}, "
                    f"but only found {len(same_val)} same-value partners"
                )
            for partner in random.sample(same_val, pos_per_pair):
                queries.append(
                    Query(
                        subject=f"{entity['name']} {partner['name']}",
                        relation=rel,
                        answer="yes",
                        query_type="composition",
                        held_out=False,
                    )
                )

            different_values = [candidate_val for candidate_val in values_by_attr[attr] if candidate_val != val]
            negative_candidates = [
                candidate
                for candidate_val in different_values
                for candidate in value_to_entities[(attr, candidate_val)]
            ]
            if len(negative_candidates) < neg_per_pair:
                raise ValueError(
                    f"Need {neg_per_pair} no examples for {entity['name']} / {rel}, "
                    f"but only found {len(negative_candidates)} different-value partners"
                )
            for neg_partner in random.sample(negative_candidates, neg_per_pair):
                queries.append(
                    Query(
                        subject=f"{entity['name']} {neg_partner['name']}",
                        relation=rel,
                        answer="no",
                        query_type="composition",
                        held_out=False,
                    )
                )

    return queries


def build_dataset(n_entities=1500, seed=32, comp_train_frac=0.5, pos_per_pair=1, neg_per_pair=1):
    random.seed(seed)

    entities = generate_entities(n_entities, seed)
    shuffled = entities[:]
    random.shuffle(shuffled)

    composition_queries = make_composition_queries(
        shuffled,
        pos_per_pair=pos_per_pair,
        neg_per_pair=neg_per_pair,
        seed=seed,
    )
    #random.shuffle(composition_queries)
    n_seen = int(comp_train_frac * len(composition_queries))
    composition_train = composition_queries[:n_seen]
    composition_held_out = composition_queries[n_seen:]

    train_queries = make_extraction_queries(entities) + composition_train
    val_queries = composition_held_out
    test_queries = train_queries + composition_held_out

    for q in composition_held_out:
        q.held_out = True

    n = len(entities)
    R = len(ALL_RELATIONS)
    print("\nDataset")
    print(f"  Entities (N):                    {n}")
    print(f"  Relations (R):                   {R}  ->  N*R = {n * R}")
    print(f"  Extraction relations:            {EXTRACTION_RELATIONS}")
    print(f"  Composition relations:           {COMPOSITION_RELATIONS}")
    print(f"  Composition examples per pair:   {pos_per_pair} yes + {neg_per_pair} no")
    print(f"  Training queries:                {len(train_queries)}")
    print(f"  Validation queries:              {len(val_queries)}")
    print(f"  Test queries (N*R):              {len(test_queries)}")

    return entities, train_queries, val_queries, test_queries


def save_dataset(entities, train_queries, val_queries, test_queries, path="dataset_composition1000.json"):
    def q2d(q):
        return {"subject": q.subject, "relation": q.relation, "answer": q.answer,
                "query_type": q.query_type, "held_out": q.held_out}
    with open(path, "w") as f:
        json.dump({
            "entities":       entities,
            "train_queries":  [q2d(q) for q in train_queries],
            "val_queries":    [q2d(q) for q in val_queries],
            "test_queries":   [q2d(q) for q in test_queries],

        }, f, indent=2)
    print(f"  Saved → {path}")

def load_dataset(path="dataset_composition1000.json"):
    with open(path) as f:
        data = json.load(f)
    def d2q(d):
        return Query(subject=d["subject"], relation=d["relation"], answer=d["answer"],
                     query_type=d["query_type"], held_out=d["held_out"])
    return (
        data["entities"],
        [d2q(d) for d in data["train_queries"]],
        [d2q(d) for d in data["val_queries"]],
        [d2q(d) for d in data["test_queries"]]
    )

import torch
from torch.utils.data import Dataset, DataLoader

class QueryDataset(Dataset):
    def __init__(self, queries, tokenizer):
        self.seqs = []
        o_id = tokenizer.convert_tokens_to_ids("<O>")
        
        for q in sorted(queries, key=lambda q: (q.subject, q.relation)):
            full = tokenizer.encode(q.sequence)
           
            x = torch.tensor(full[:-1], dtype=torch.long)
            y = torch.tensor(full[1:], dtype=torch.long)

            mask = torch.zeros(len(x), dtype = torch.bool)
            for i, tok in enumerate(x.tolist()):
                if tok == o_id:
                    mask[i] = True
                    break
            self.seqs.append((x,y,mask))
        

    def __len__(self):        return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i]

def collate_fn(batch):
    inputs  = pad_sequence([b[0] for b in batch], batch_first=True, padding_value=0)
    targets = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=0)
    masks   = pad_sequence([b[2] for b in batch], batch_first=True, padding_value=False)

    return inputs, targets, masks

if __name__ == "__main__":
    entities, train_q, val_q, test_q = build_dataset(n_entities=4000) 
    e = entities[0]
    print(f"\nSample entity:\n  {json.dumps(e, indent=4)}")
    print(f"\nTraining sequences for {e['name']}:")
    for q in [q for q in train_q if q.query_type == "composition"]:
        print(f" {q.sequence}")
    print(f"\nTest sequences for {e['name']}:")
    for q in [q for q in test_q if q.subject == e["name"]][:14]:
        print(f"  [{q.query_type:11}] {q.sequence}")
    save_dataset(entities, train_q, val_q, test_q) 

 
def get_dataloaders(train_queries, tokenizer, batch_size=32, seed=42):
    ds = QueryDataset(train_queries, tokenizer)

    g = torch.Generator()
    g.manual_seed(seed)

    tr_dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    print(f"  Train sequences: {len(ds)}  ({len(tr_dl)} batches)")
    return tr_dl
 
 