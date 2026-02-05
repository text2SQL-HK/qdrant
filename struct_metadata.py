from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, List, Tuple

DEFAULT_TABLES_JSON = Path("/tmp/workspace/LLM_project/output/tables1223.json")
DEFAULT_EMBED_JSON = Path("/tmp/workspace/property_graph/embed.json")
DEFAULT_DB_PATH = Path("/tmp/workspace/qdrant/metadata.sqlite")


def _load_tables_json(path: Path, db_id: str | None) -> dict:
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    tables_data = loaded if isinstance(loaded, list) else [loaded]
    if db_id:
        db_info = next((item for item in tables_data if item.get("db_id") == db_id), None)
        if not db_info:
            raise ValueError(f"DB ID '{db_id}' not found in {path}")
        return db_info
    if len(tables_data) != 1:
        raise ValueError("Multiple DBs found. Provide --db-id to select one.")
    return tables_data[0]


def _load_embed_json(path: Path, db_id: str | None) -> dict:
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if isinstance(loaded, list):
        embed_data = {item["db_id"]: item for item in loaded if isinstance(item, dict) and "db_id" in item}
    elif isinstance(loaded, dict):
        embed_data = loaded if "db_id" not in loaded else {loaded["db_id"]: loaded}
    else:
        embed_data = {}
    if not embed_data:
        raise ValueError(f"Unsupported embed.json format: {path}")
    if db_id:
        if db_id not in embed_data:
            raise ValueError(f"DB ID '{db_id}' not found in {path}")
        return embed_data[db_id]
    if len(embed_data) != 1:
        raise ValueError("Multiple DBs found in embed.json. Provide --db-id to select one.")
    return next(iter(embed_data.values()))


def _build_desc_str_map(embed_db_info: dict) -> dict[tuple[str, str], str]:
    desc_map: dict[tuple[str, str], str] = {}
    for table_name, table_info in embed_db_info.items():
        cols = table_info.get("col", {})
        for col_name, col_info in cols.items():
            desc_str = col_info.get("desc_str")
            if desc_str:
                desc_map[(table_name, col_name)] = desc_str
    return desc_map


def _iter_table_rows(db_info: dict) -> Iterable[Tuple[str, str]]:
    table_names_orig: List[str] = db_info["table_names_original"]
    table_desc: List[str] = db_info["table_names"]
    if len(table_names_orig) != len(table_desc):
        raise ValueError("tables.json: 'table_names_original' and 'table_names' lengths differ.")
    for idx, table_name in enumerate(table_names_orig):
        yield table_name, table_desc[idx]


def _iter_column_rows(
    db_info: dict,
    desc_map: dict[tuple[str, str], str],
) -> Iterable[Tuple[str, str, str]]:
    table_names_orig: List[str] = db_info["table_names_original"]
    cols_orig: List[List[Any]] = db_info["column_names_original"]
    cols_desc: List[List[Any]] = db_info["column_names"]
    if len(cols_orig) != len(cols_desc):
        raise ValueError("tables.json: 'column_names_original' and 'column_names' lengths differ.")
    for idx, (t_idx, col_name) in enumerate(cols_orig):
        if t_idx is None or t_idx < 0 or col_name == "*":
            continue
        table_name = table_names_orig[t_idx]
        desc_str = desc_map.get((table_name, col_name))
        if not desc_str:
            continue
        yield table_name, col_name, desc_str


def build_sqlite(db_path: Path, db_info: dict, desc_map: dict[tuple[str, str], str]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(
            """
            DROP TABLE IF EXISTS column_metadata;
            DROP TABLE IF EXISTS table_metadata;
            CREATE TABLE table_metadata (
                table_name TEXT PRIMARY KEY,
                table_desc TEXT NOT NULL
            );
            CREATE TABLE column_metadata (
                table_name  TEXT NOT NULL,
                column_name TEXT NOT NULL,
                column_desc TEXT NOT NULL,

                PRIMARY KEY (table_name, column_name),
                FOREIGN KEY (table_name)
                    REFERENCES table_metadata(table_name)
                    ON DELETE CASCADE
            );
            """
        )
        conn.executemany(
            "INSERT INTO table_metadata (table_name, table_desc) VALUES (?, ?);",
            list(_iter_table_rows(db_info)),
        )
        conn.executemany(
            "INSERT INTO column_metadata (table_name, column_name, column_desc) VALUES (?, ?, ?);",
            list(_iter_column_rows(db_info, desc_map)),
        )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQLite metadata from tables1223.json + embed.json")
    parser.add_argument("--tables-json", type=Path, default=DEFAULT_TABLES_JSON)
    parser.add_argument("--embed-json", type=Path, default=DEFAULT_EMBED_JSON)
    parser.add_argument("--db-id", default=None, help="DB ID to select when multiple entries exist.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    db_info = _load_tables_json(args.tables_json, args.db_id)
    embed_info = _load_embed_json(args.embed_json, args.db_id)
    desc_map = _build_desc_str_map(embed_info)
    build_sqlite(args.db_path, db_info, desc_map)
    print(f"SQLite metadata created at: {args.db_path}")


if __name__ == "__main__":
    main()
