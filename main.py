import os
import re
import json
import time
import numpy as np
import pandas as pd
import chromadb
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from groq import Groq

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
RAG_CSV_PATH    = os.environ.get("RAG_CSV_PATH",    "rag_documents.csv")
EMBEDDINGS_PATH = os.environ.get("EMBEDDINGS_PATH", "embeddings.npy")
CHROMA_DIR      = os.environ.get("CHROMA_DIR",      "./chroma_db")
COLLECTION_NAME = "anomaly_logs"
NPY_PATH = os.environ.get("NPY_PATH", "embeddings.npy")

# ─────────────────────────────────────────────────────────────────────────────
# EVENT DESCRIPTIONS
# ─────────────────────────────────────────────────────────────────────────────
EVENT_DESCRIPTIONS = {
    "E1":  "A DataNode received a request to store a block that already exists — duplicate write or stale reference.",
    "E2":  "Block checksum verification passed — data written is confirmed intact.",
    "E3":  "A DataNode successfully served a block read request.",
    "E4":  "An exception occurred while serving a block — read or transfer failed.",
    "E5":  "A DataNode started receiving a new block — start of a pipeline write.",
    "E6":  "A DataNode finished receiving a full block — complete transfer confirmed.",
    "E7":  "An exception was thrown during a block write operation.",
    "E8":  "The PacketResponder thread was interrupted — unexpected pipeline disruption.",
    "E9":  "A DataNode finished receiving a block of a known size — successful replica receipt.",
    "E10": "The PacketResponder thread threw an unhandled exception.",
    "E11": "The PacketResponder thread for a block terminated — normal end or failure.",
    "E12": "Exception while writing block to a mirror DataNode — replication pipeline failed.",
    "E13": "DataNode received an empty packet — heartbeat or end-of-stream signal.",
    "E14": "Exception inside receiveBlock handler — block write could not complete.",
    "E15": "NameNode adjusted block offset metadata — recovery or corruption repair.",
    "E16": "Block successfully transferred to another DataNode.",
    "E17": "Block transfer to target DataNode failed — re-replication unsuccessful.",
    "E18": "NameNode instructed DataNode to start background block copy thread.",
    "E19": "Block file reopened for appending.",
    "E20": "Delete failed — block metadata not found in DataNode volume map (orphaned block).",
    "E21": "DataNode deleted a block file from local disk — triggered by NameNode invalidation.",
    "E22": "NameNode allocated a new block ID — start of new block creation.",
    "E23": "NameNode added block to invalidation set — scheduled for deletion.",
    "E24": "NameNode removed block from replication queue — no longer belongs to any file.",
    "E25": "NameNode instructed DataNode to replicate block — under-replication detected.",
    "E26": "NameNode updated block map after DataNode reported successful storage.",
    "E27": "Redundant block storage report received — duplicate reporting.",
    "E28": "Block report for unknown file received — block is orphaned.",
    "E29": "Replication request timed out — target DataNode did not complete copy in time.",
}

FALLBACK_MITIGATIONS = {
    "duplicate_pattern": {
        "high":   ["Check DataNode pipeline threads for race conditions causing duplicate block writes.",
                   "Enable idempotency checks on NameNode to reject duplicate block registrations."],
        "medium": ["Review client retry config — reduce max retries or add backoff jitter.",
                   "Monitor DataNode network throughput for intermittent failures."],
        "low":    ["Audit HDFS client version for known duplicate-write bugs."],
    },
    "repetition": {
        "high":   ["Inspect DataNode for stuck PacketResponder thread and restart if confirmed.",
                   "Check for network packet loss between pipeline DataNodes."],
        "medium": ["Review NameNode RPC logs for timeout patterns triggering re-sends.",
                   "Verify DataNode JVM heap — GC pauses can stall pipeline and trigger retries."],
        "low":    ["Increase pipeline write timeout thresholds to reduce false retry triggers."],
    },
    "missing_events": {
        "high":   ["Inspect DataNode that aborted pipeline — check for disk errors or OOM.",
                   "Verify block replication in NameNode — trigger re-replication if under-replicated."],
        "medium": ["Review DataNode stderr logs around block timestamp for crash evidence.",
                   "Check network stability between pipeline nodes for partial disconnects."],
        "low":    ["Add HDFS block scanner runs to detect and repair corrupted replicas."],
    },
    "high_latency": {
        "high":   ["Profile DataNode disk I/O at write time — check for saturation or slow disks.",
                   "Check network bandwidth between NameNode and DataNodes during latency window."],
        "medium": ["Review JVM GC logs on affected DataNode for stop-the-world pauses.",
                   "Verify DataNode CPU is not contended by co-located processes."],
        "low":    ["Move high-throughput workloads to dedicated DataNodes to reduce latency variance."],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────
documents    = []
all_metadata = []
block_lookup = {}
collection   = None
groq_client  = None
# ONNX session for query-time embedding (tiny memory vs full torch)
onnx_session     = None
tokenizer        = None


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global documents, all_metadata, block_lookup, collection, groq_client
    global onnx_session, tokenizer

    print("[startup] Loading RAG documents...")
    rag_df    = pd.read_csv(RAG_CSV_PATH)
    documents = rag_df["RAG_Document"].tolist()
    print(f"[startup] {len(documents)} documents loaded.")

    all_metadata = [parse_metadata_from_doc(doc) for doc in documents]

    print("[startup] Loading pre-computed embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    print(f"[startup] Embeddings shape: {embeddings.shape}")

    print("[startup] Building ChromaDB index (in-memory)...")
    chroma_client = chromadb.EphemeralClient()   # no disk — saves memory
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    BATCH = 500
    for start in range(0, len(documents), BATCH):
        end = min(start + BATCH, len(documents))
        collection.add(
            documents  = documents[start:end],
            embeddings = embeddings[start:end].tolist(),
            ids        = [str(i) for i in range(start, end)],
            metadatas  = all_metadata[start:end],
        )
    print(f"[startup] Indexed {collection.count()} vectors.")

    # Load ONNX model for query embedding — ~80MB vs ~300MB for torch
    print("[startup] Loading ONNX embedding model...")
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
    tokenizer    = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    onnx_session = ORTModelForFeatureExtraction.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", export=True
    )
    print("[startup] ONNX model ready.")

    block_lookup = {
        m["block_id"]: (documents[i], m)
        for i, m in enumerate(all_metadata)
    }

    groq_client = Groq(api_key=GROQ_API_KEY)
    print("[startup] Ready.")
    yield
    print("[shutdown] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def parse_metadata_from_doc(doc: str) -> dict:
    block_id_m = re.search(r"Block ID\s*:\s*(\S+)", doc)
    atypes_m   = re.search(r"Anomaly Type\(s\)\s*:\s*(.+)", doc)
    latency_m  = re.search(r"Latency\s*:\s*(\d+)", doc)
    total_m    = re.search(r"Total Events\s*:\s*(\d+)", doc)
    full_types   = atypes_m.group(1).strip() if atypes_m else "unknown"
    primary_type = full_types.split(",")[0].strip()
    return {
        "block_id":      block_id_m.group(1).strip() if block_id_m else "unknown",
        "anomaly_types": full_types,
        "primary_type":  primary_type,
        "latency":       int(latency_m.group(1)) if latency_m else 0,
        "total_events":  int(total_m.group(1))   if total_m   else 0,
    }


def embed_query(text: str) -> list:
    """Embed a single query string using ONNX — no torch required."""
    import torch
    inputs  = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=256, padding=True)
    outputs = onnx_session(**inputs)
    # Mean pooling
    token_embeddings = outputs.last_hidden_state
    attention_mask   = inputs["attention_mask"]
    mask_expanded    = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled           = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
    vec              = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return vec.detach().numpy().tolist()


def compress_historical_case(doc, block_id, latency, distance) -> str:
    seq_m   = re.search(r"Event Sequence\s*:\s*(.+)", doc)
    atype_m = re.search(r"Anomaly Type\(s\)\s*:\s*(.+)", doc)
    total_m = re.search(r"Total Events\s*:\s*(\d+)", doc)
    seq     = seq_m.group(1).strip()   if seq_m   else ""
    atype   = atype_m.group(1).strip() if atype_m else "unknown"
    total   = total_m.group(1)         if total_m else "?"
    tokens  = [t.strip() for t in seq.split("->") if t.strip()]
    short   = " -> ".join(tokens[:15])
    if len(tokens) > 15:
        short += f" (+{len(tokens)-15} more)"
    return (f"Block: {block_id} | {atype} | {latency}ms | dist={distance:.3f} | events={total}\n"
            f"Seq: {short}")


def build_query_from_doc(doc, meta) -> str:
    seq_m = re.search(r"Event Sequence\s*:\s*(.+)", doc)
    seq   = seq_m.group(1).strip() if seq_m else ""
    return (f"HDFS anomaly: {meta.get('anomaly_types','?')}. "
            f"Latency: {meta.get('latency',0)}ms. Events: {meta.get('total_events',0)}. "
            f"Seq: {seq}")


def build_event_context(doc) -> str:
    seq_m = re.search(r"Event Sequence\s*:\s*(.+)", doc)
    if not seq_m:
        return "No event sequence."
    ids = list(dict.fromkeys(re.findall(r"E\d+", seq_m.group(1))))
    return "\n".join(f"{e}: {EVENT_DESCRIPTIONS.get(e,'Unknown')}" for e in ids)


def retrieve_similar(query, k=3, anomaly_type_filter=None):
    qemb  = embed_query(query)
    where = {"primary_type": {"$eq": anomaly_type_filter}} if anomaly_type_filter else None
    res   = collection.query(query_embeddings=qemb, n_results=k, where=where)
    return [
        {"document": doc, "block_id": m.get("block_id","?"),
         "primary_type": m.get("primary_type","?"),
         "latency": m.get("latency", 0), "distance": d}
        for doc, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def repair_json(s: str) -> dict:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"(?<![\w])'([^']*)'(?![\w])", r'"\1"', s)
    result = []; in_string = False; i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and in_string:
            result.append(c); i += 1
            if i < len(s): result.append(s[i])
            i += 1; continue
        if c == '"':
            if not in_string:
                in_string = True; result.append(c)
            else:
                rest   = s[i+1:].lstrip()
                closes = (not rest) or rest[0] in (":", ",", "}", "]")
                if closes or i == len(s)-1:
                    in_string = False; result.append(c)
                else:
                    result.append('\\"')
            i += 1; continue
        result.append(c); i += 1
    s = "".join(result)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        if in_string: s += '"'
        depth = []
        for ch in s:
            if ch in ("{","["): depth.append("}" if ch=="{" else "]")
            elif ch in ("}","]") and depth: depth.pop()
        s += "".join(reversed(depth))
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return json.loads(s)


def generate_root_cause(query_doc, hist_ctx, query_meta, similar_docs) -> dict:
    total_events = query_meta.get("total_events", "?")
    latency_ms   = query_meta.get("latency", "?")
    event_ref    = build_event_context(query_doc)

    prompt = f"""QUERY BLOCK:
{query_doc}

EVENT REFERENCE:
{event_ref}

HISTORICAL CASES:
{hist_ctx}

Return JSON in EXACTLY this field order:
{{
  "block_id": "{query_meta['block_id']}",
  "anomaly_type": "copy first type from Anomaly Type(s) exactly",
  "confidence": 0.85,
  "confidence_label": "high",
  "mitigation_steps": {{
    "high": ["urgent action 1", "urgent action 2"],
    "medium": ["follow-up 1", "follow-up 2"],
    "low": ["improvement"]
  }},
  "summary": "THIS BLOCK HAS {total_events} EVENTS AND {latency_ms}ms LATENCY. 1-2 sentences with these exact numbers.",
  "root_cause": "mechanism first, then cite event IDs as evidence",
  "comparison_to_historical": "one sentence with one specific number",
  "event_explanations": {{"E5": "copy from EVENT REFERENCE"}}
}}

RULES: mitigation all tiers required. anomaly_type from QUERY BLOCK only. JSON only."""

    MAX_RETRIES = 3
    response = None; last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are an HDFS expert. Return ONLY valid JSON."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2, max_tokens=1500,
            )
            break
        except Exception as e:
            last_error = e; es = str(e)
            if "413" in es and attempt < MAX_RETRIES-1:
                compressed = "\n".join(
                    f"Case {i+1}: {r['block_id']} | {r['primary_type']} | {r['latency']}ms"
                    for i, r in enumerate(similar_docs)
                )
                prompt = prompt.replace(hist_ctx, compressed)
                continue
            if "429" in es and attempt < MAX_RETRIES-1:
                time.sleep(5*(attempt+1)); continue
            break

    if response is None:
        return {"error": str(last_error), "block_id": query_meta.get("block_id","?")}

    try:
        content = response.choices[0].message.content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        try:
            parsed = repair_json(content)
        except Exception as e:
            print(f"[PARSE ERROR] {query_meta.get('block_id','?')} | {e} | {content[:200]}")
            parsed = {
                "block_id": query_meta.get("block_id","?"), "summary": "Parse failed.",
                "root_cause": content[:300], "comparison_to_historical": "",
                "event_explanations": {}, "anomaly_type": query_meta.get("primary_type","unknown"),
                "confidence": 0.0, "confidence_label": "low",
                "mitigation_steps": {"high":[],"medium":[],"low":[]}, "parse_error": True,
            }

        valid_types = {"duplicate_pattern","repetition","missing_events","high_latency"}
        if parsed.get("anomaly_type") not in valid_types:
            parsed["anomaly_type"] = query_meta.get("primary_type","unknown")

        atype    = parsed.get("anomaly_type", query_meta.get("primary_type","duplicate_pattern"))
        fallback = FALLBACK_MITIGATIONS.get(atype, FALLBACK_MITIGATIONS["duplicate_pattern"])
        mit      = parsed.get("mitigation_steps")
        if not isinstance(mit, dict):
            parsed["mitigation_steps"] = fallback
        else:
            for tier in ("high","medium","low"):
                cleaned = [x for x in (mit.get(tier) or []) if x and str(x).strip()]
                mit[tier] = cleaned if cleaned else fallback[tier]

        cth = parsed.get("comparison_to_historical","")
        if isinstance(cth,(list,dict)) or (isinstance(cth,str) and cth.strip().startswith("[")):
            avg = int(sum(r["latency"] for r in similar_docs)/max(len(similar_docs),1))
            parsed["comparison_to_historical"] = (
                f"Similar pattern to retrieved cases (avg historical: {avg}ms vs this block: {latency_ms}ms)."
            )

        return parsed
    except Exception as e:
        return {"error": str(e), "block_id": query_meta.get("block_id","?")}


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="HDFS Anomaly Root Cause API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


class AnalyzeRequest(BaseModel):
    query:               Optional[str] = None
    block_id:            Optional[str] = None
    anomaly_type_filter: Optional[str] = None
    k:                   Optional[int] = 3


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "blocks": "/blocks"}

@app.get("/health")
def health():
    return {"status": "ok", "documents": len(documents),
            "vectors": collection.count() if collection else 0}

@app.get("/blocks")
def list_blocks(limit: int = 20, anomaly_type: Optional[str] = None):
    blocks = [
        {"block_id": m["block_id"], "primary_type": m["primary_type"], "latency_ms": m["latency"]}
        for m in all_metadata
        if not anomaly_type or m["primary_type"] == anomaly_type
    ]
    return {"total": len(blocks), "blocks": blocks[:limit]}

@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    if not request.query and not request.block_id:
        raise HTTPException(400, "Provide 'query' or 'block_id'.")

    if request.block_id:
        if request.block_id not in block_lookup:
            raise HTTPException(404, f"block_id '{request.block_id}' not found.")
        q_doc, q_meta = block_lookup[request.block_id]
        query_str     = build_query_from_doc(q_doc, q_meta)
        atype_filter  = request.anomaly_type_filter or q_meta["primary_type"]
    else:
        q_doc        = request.query
        q_meta       = {"block_id": "external_query",
                        "primary_type": request.anomaly_type_filter or "unknown",
                        "anomaly_types": request.anomaly_type_filter or "unknown",
                        "latency": 0, "total_events": 0}
        query_str    = request.query
        atype_filter = request.anomaly_type_filter

    similar = retrieve_similar(query_str, k=request.k, anomaly_type_filter=atype_filter)
    hist_ctx = "\n\n".join([
        f"Case {i+1}\n" + compress_historical_case(r["document"],r["block_id"],r["latency"],r["distance"])
        for i, r in enumerate(similar)
        if r["block_id"] != q_meta.get("block_id","")
    ])

    result = generate_root_cause(q_doc, hist_ctx, q_meta, similar)
    result["retrieved_block_ids"] = [r["block_id"] for r in similar]
    result["query_block_id"]      = q_meta.get("block_id","external_query")
    return result
