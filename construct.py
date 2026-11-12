from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import json
import sqlite3
from openai import OpenAI
from typing import Dict, List, Any, Optional
import uuid
import os
import csv
import requests

date = "1223"
RERANK_URL = os.environ.get("RERANK_URL", "http://localhost:8002/rerank")
TABLE_JSON_PATH = os.environ.get("TABLES_JSON_PATH") or "data/tables.json"
METADATA_DB_PATH = os.environ.get("METADATA_DB_PATH") or "data/metadata.sqlite"

openai_client = OpenAI(
    base_url=os.environ.get("EMBEDDING_BASE_URL", "http://localhost:8001/v1"), api_key=os.environ.get("EMBEDDING_API_KEY") or "EMPTY"
)
output_path = os.environ.get("EMBED_OUTPUT_PATH") or "data/embed.json"

COLUMN_COLLECTION_NAME = "column_collection"
TABLE_COLLECTION_NAME = "table_collection"
instruction = """
You are selecting database columns that are likely required to write a correct SQL query for the user question.

Return "yes" ONLY if the column is directly needed for:
1) filtering conditions (WHERE),
2) grouping/aggregation (GROUP BY / HAVING),
3) joins/keys (JOIN),
4) selected outputs (SELECT),
5) sorting (ORDER BY)

Return "no" if the column is merely background information, redundant, or only loosely related.
Be conservative: prefer "no" unless the column is clearly useful for producing SQL.
Take care of [Context].
[Context]
{context}
"""

def rerank(query:str, columns:List[str], context):
    payload = {
        "query": query,
        "instruction": instruction.format(context=context),
        "documents": columns
    }
    resp = requests.post(RERANK_URL, json=payload)
    resp.raise_for_status()
    return (resp.json())["results"]

def embed(input: str) -> List:
    response = openai_client.embeddings.create(
        model = "/workspace/models/Qwen3-Embedding-0.6B",
        input= input
    )
    return response.data[0].embedding

def generate_data(tables_json_path, db_id: str):
    def flatten_json_examples(col_entry: Any) -> List[Any]:
        if col_entry is None:
            return []
        if isinstance(col_entry, dict):
            if "value" in col_entry:
                col_entry = col_entry["value"]
            elif "values" in col_entry:
                col_entry = col_entry["values"]
            elif "examples" in col_entry:
                col_entry = col_entry["examples"]
        examples: List[Any] = []
        if isinstance(col_entry, list):
            if col_entry and all(isinstance(x, list) for x in col_entry):
                for j, inner in enumerate(col_entry):
                    if j == 0:
                        continue
                    examples.extend(inner)
            else:
                examples = list(col_entry)
        else:
            examples = [col_entry]
        cleaned: List[Any] = []
        seen = set()
        for item in examples:
            if item is None:
                continue
            hashable = (
                json.dumps(item, ensure_ascii=False, sort_keys=True)
                if isinstance(item, (dict, list))
                else item
            )
            key = (type(item), hashable)
            if key not in seen:
                seen.add(key)
                cleaned.append(item)
        return cleaned[:5]

    def is_dummy_col(t_idx: Any, col_name: Any) -> bool:
        return (t_idx is None) or (t_idx < 0) or (col_name == "*")

    def get_col_examples(values_block: Any, col_idx: int) -> List[Any]:
        if values_block is None:
            return []
        if isinstance(values_block, list):
            col_entry = values_block[col_idx] if col_idx < len(values_block) else None
        elif isinstance(values_block, dict):
            col_entry = values_block.get(str(col_idx), values_block.get(col_idx))
        else:
            col_entry = None
        try:
            return flatten_json_examples(col_entry)
        except Exception:
            return []

    with open(tables_json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    tables_data = loaded if isinstance(loaded, list) else [loaded]
    db_info = next((item for item in tables_data if item.get("db_id") == db_id), None)
    if not db_info:
        raise ValueError(f"DB ID '{db_id}' not found in {tables_json_path}")

    table_names_orig: List[str] = db_info["table_names_original"]
    table_desc: List[str] = db_info["table_names"]
    cols_orig: List[List[Any]] = db_info["column_names_original"]
    cols_desc: List[List[Any]] = db_info["column_names"]
    cols_type: List[List[Any]] = db_info["column_types"]

    values_block: Any = db_info.get("values")

    if len(cols_orig) != len(cols_desc):
        raise ValueError(
            "tables.json: 'column_names_original' and 'column_names' lengths differ."
        )
    
    table_to_cols: List[List[Tuple[int, str, str]]] = [[] for _ in table_names_orig]
    for idx, (t_idx, col_name_actual) in enumerate(cols_orig):
        if is_dummy_col(t_idx, col_name_actual):
            continue
        _, maybe_desc_text = cols_desc[idx]
        col_type = cols_type[idx]
        col_desc_text = (maybe_desc_text or col_name_actual.replace("_", " ")).strip()
        table_to_cols[t_idx].append((idx, col_name_actual, col_desc_text, col_type))

    result = {db_id: {}}
    table_dict = result[db_id]
    for t_idx, table_name in enumerate(table_names_orig):
        table_name_desc = table_desc[t_idx]
        table_dict[table_name] = { "desc" : table_name_desc }
        table = table_dict[table_name]
        table["embed_desc"] = embed(table_name_desc)
        table["col"] = {}
        for col_idx, col_name_actual, col_desc, col_type in table_to_cols[t_idx]:
            examples = get_col_examples(values_block, col_idx)
            if examples == []:
                continue
            table["col"][col_name_actual] = { "desc": col_desc }
            col = table["col"][col_name_actual]
            col["value_example"] = f"{examples}"
            desc_str = f"{col_name_actual}은(는) {col_desc}을(를) 의미한다. {col_type} 형식이고, 예시 값으로는 {examples} 등이 있다."
            col["embed_desc"] = embed(desc_str)
            col["type"] = col_type
            col["desc_str"] = desc_str
            
    return result

if not os.path.exists(output_path):
    result = generate_data(TABLE_JSON_PATH, "EXAMPLE_DATABASE")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

else:
    with open(output_path, "r", encoding="utf-8") as f:
        result = json.load(f)

def upsert_columns(client, result):
    for db_id, tables in result.items():
        for (table_name, meta) in tables.items():
            for (column_name, column) in meta["col"].items():
                point_key = f"{db_id}::{table_name}::{column_name}"
                client.upsert(
                    collection_name=COLUMN_COLLECTION_NAME,
                    points=[
                        {
                            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, point_key)),
                            "vector": column["embed_desc"],
                            "payload": {
                                "db_id": db_id,
                                "table_name": table_name,
                                "column_name": column_name,
                                "desc": column["desc_str"]
                            },
                        }
                    ],
                )

def upsert_tables(client, result):
    for db_id, tables in result.items():
        for (table_name, meta) in tables.items():
            point_key = f"{db_id}::{table_name}"
            client.upsert(
                collection_name=TABLE_COLLECTION_NAME,
                points=[
                    {
                        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, point_key)),
                        "vector": meta["embed_desc"],
                        "payload": {
                            "key": point_key,
                            "db_id": db_id,
                            "table_name": table_name,
                            "desc": meta["desc"]
                        },
                    }
                ],
            )


def build_desc_str_from_metadata(
    conn: sqlite3.Connection,
    table_names: List[str],
    query: str,
    context: str = "",
    top_k: int = 7,
) -> str:
    desc_parts: List[str] = []
    for table_name in table_names:
        row = conn.execute(
            "SELECT table_desc FROM table_metadata WHERE table_name = ?;",
            (table_name,),
        ).fetchone()
        if row is None:
            continue
        table_desc = row[0]
        desc_parts.append(f"#Table: {table_name}\n{table_desc}\nColumns\n")
        desc_parts.append("[\n")
        col_rows = conn.execute(
            "SELECT column_desc FROM column_metadata WHERE table_name = ? ORDER BY column_name;",
            (table_name,),
        ).fetchall()
        col_desc_list = [column_desc for (column_desc,) in col_rows]
        if col_desc_list:
            ranked = rerank(query, col_desc_list, context)
            limit = min(top_k, len(ranked))
            for i in range(limit):
                desc_parts.append(f" {ranked[i]['document']}\n")
        desc_parts.append("]\n")
    return "".join(desc_parts)


client = QdrantClient(host="localhost", port=6333)
if not client.collection_exists(collection_name=COLUMN_COLLECTION_NAME):
    client.create_collection(
        collection_name=COLUMN_COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.DOT),
    )
print(f"Collection '{COLUMN_COLLECTION_NAME}' created with Dot Product distance.")

if not client.collection_exists(collection_name=TABLE_COLLECTION_NAME):
    client.create_collection(
        collection_name=TABLE_COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.DOT),
    )
print(f"Collection '{TABLE_COLLECTION_NAME}' created with Dot Product distance.")

upsert_columns(client, result)
upsert_tables(client, result)

query = "금산에서 있었던 불가동 이력 가져와줘"
query_vector = embed(query)

column_search_result = client.query_points(
    collection_name=COLUMN_COLLECTION_NAME,
    query=query_vector,
    limit=30
)

table_search_result = client.query_points(
    collection_name=TABLE_COLLECTION_NAME,
    query=query_vector,
    limit=5
)

print("=== Column Search Results ===")
column_rows = [
    {
        "id": p.id,
        "payload": p.payload,
        "score": p.score
    }
    for p in column_search_result.points
]
print(column_rows)
print("=== Table Search Results ===")
table_rows = [
    {
        "id": p.id,
        "payload": p.payload,
        "score": p.score
    }
    for p in table_search_result.points
]
print(table_rows)

db_id_set = set()
score_dict = {}
db_table_dict = {}

for row in column_rows:
    db_id_set.add(row["payload"]["db_id"])
for row in table_rows:
    db_id_set.add(row["payload"]["db_id"])

for db_id in db_id_set:
    db_table_dict[db_id] = set()

for row in column_rows:
    db_table_dict[row["payload"]["db_id"]].add(row["payload"]["table_name"])
for row in table_rows:
    db_table_dict[row["payload"]["db_id"]].add(row["payload"]["table_name"])

for db_id, table_set in db_table_dict.items():
    score_list = []
    for table_name in table_set:
        score_list.append({"table_name": table_name, "highest_column_score": 0.0, "table_score": 0.0, "column_cnt": 0})
    score_dict[db_id] = score_list

for row in column_rows:
    db_id = row["payload"]["db_id"]
    table_name = row["payload"]["table_name"]
    for entry in score_dict[db_id]:
        if entry["table_name"] == table_name:
            entry["column_cnt"] += 1
            if row["score"] > entry["highest_column_score"]:
                entry["highest_column_score"] = row["score"]
                

for row in table_rows:
    db_id = row["payload"]["db_id"]
    table_name = row["payload"]["table_name"]
    for entry in score_dict[db_id]:
        if entry["table_name"] == table_name:
            entry["table_score"] = row["score"]

for db_id in score_dict:
    for entry in score_dict[db_id]:
        norm_cnt = min(entry["column_cnt"], 5)/5.0
        entry["final_score"] = (
            norm_cnt * 0.5 +
            entry["highest_column_score"] * 0.3 +
            entry["table_score"] * 0.5
        )

for db_id in score_dict:
    score_dict[db_id] = sorted(
        score_dict[db_id],
        key=lambda x: x["final_score"],
        reverse=True
    )

top_tables_by_db = {
    db_id: [entry["table_name"] for entry in entries[:2]]
    for db_id, entries in score_dict.items()
}

with sqlite3.connect(METADATA_DB_PATH) as conn:
    for db_id, table_names in top_tables_by_db.items():
        desc_str = build_desc_str_from_metadata(conn, table_names, query, context="")
        print("=== Related Schema ===")
        print(desc_str)
