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
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (all sensitive values come from Render environment variables)
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
RAG_CSV_PATH   = os.environ.get("RAG_CSV_PATH", "rag_documents.csv")
CHROMA_DIR     = os.environ.get("CHROMA_DIR", "./chroma_db")
EMBED_MODEL    = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "anomaly_logs"
NPY_PATH = os.environ.get("NPY_PATH", "embeddings.npy")

# ─────────────────────────────────────────────────────────────────────────────
# EVENT DESCRIPTIONS
# ─────────────────────────────────────────────────────────────────────────────
EVENT_DESCRIPTIONS = {
    "E1":  "A DataNode received a request to store a block that already exists — duplicate write or stale reference.",
    "E2":  "Block checksum verification passed — data written is confirmed intact.",
    "E3":  "A DataNode successfully served a block read request.",
    "E4":  "An exception occurred while a DataNode was serving a block — read or transfer failed.",
    "E5":  "A DataNode started receiving a new block — normal start of a pipeline write.",
    "E6":  "A DataNode finished receiving a full block — confirms complete transfer.",
    "E7":  "An exception was thrown during a block write operation — write pipeline error.",
    "E8":  "The PacketResponder thread was interrupted — unexpected pipeline disruption.",
    "E9":  "A DataNode finished receiving a block of a known size — successful replica receipt.",
    "E10": "The PacketResponder thread threw an unhandled exception — serious acknowledgement loop error.",
    "E11": "The PacketResponder thread for a block terminated — normal end or abnormal failure.",
    "E12": "Exception while writing block to a mirror DataNode — replication pipeline failed.",
    "E13": "DataNode received an empty packet — heartbeat or end-of-stream signal.",
    "E14": "Exception inside receiveBlock handler — block write could not complete.",
    "E15": "NameNode adjusted block offset metadata — recovery or corruption repair.",
    "E16": "Block successfully transferred to another DataNode.",
    "E17": "Block transfer to target DataNode failed — re-replication unsuccessful.",
    "E18": "NameNode instructed DataNode to start background block copy thread.",
    "E19": "Block file reopened for appending.",
    "E20": "Delete failed — block metadata not found in DataNode volume map, orphaned block.",
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
                   "Check for network packet loss between pipeline DataNodes causing repeated events."],
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
# GLOBALS (populated at startup)
# ─────────────────────────────────────────────────────────────────────────────
documents     = []
all_metadata  = []
block_lookup  = {}
embed_model   = None
collection    = None
groq_client   = None


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP  — load data, build embeddings, index ChromaDB
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global documents, all_metadata, block_lookup, embed_model, collection, groq_client

    print("[startup] Loading RAG documents...")
    rag_df    = pd.read_csv(RAG_CSV_PATH)
    documents = rag_df["RAG_Document"].tolist()
    print(f"[startup] {len(documents)} documents loaded.")

    all_metadata = [parse_metadata_from_doc(doc) for doc in documents]

    print("[startup] Loading embedding model (lazy — skipping bulk encode)...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("[startup] Building ChromaDB index from pre-computed embeddings...")
    chroma_client = chromadb.Client()   # in-memory only, no persist_directory
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    NPY_PATH = os.environ.get("NPY_PATH", "embeddings.npy")
    embeddings = np.load(NPY_PATH)      # load pre-computed, ~5MB, instant
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

    block_lookup = {
        m["block_id"]: (documents[i], m)
        for i, m in enumerate(all_metadata)
    }

    groq_client = Groq(api_key=GROQ_API_KEY)
    print("[startup] Ready.")
    yield
    print("[shutdown] Goodbye.")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def parse_metadata_from_doc(doc: str) -> dict:
    block_id_m = re.search(r"Block ID\s*:\s*(\S+)", doc)
    label_m    = re.search(r"^Label\s*:\s*(.+)", doc, re.MULTILINE)
    atypes_m   = re.search(r"Anomaly Type\(s\)\s*:\s*(.+)", doc)
    latency_m  = re.search(r"Latency\s*:\s*(\d+)", doc)
    total_m    = re.search(r"Total Events\s*:\s*(\d+)", doc)
    full_types   = atypes_m.group(1).strip() if atypes_m else "unknown"
    primary_type = full_types.split(",")[0].strip()
    return {
        "block_id":      block_id_m.group(1).strip() if block_id_m else "unknown",
        "label":         label_m.group(1).strip()    if label_m    else "Fail",
        "anomaly_types": full_types,
        "primary_type":  primary_type,
        "latency":       int(latency_m.group(1))     if latency_m  else 0,
        "total_events":  int(total_m.group(1))       if total_m    else 0,
    }


def compress_historical_case(doc: str, block_id: str, latency: int, distance: float) -> str:
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


def build_query_from_doc(doc: str, meta: dict) -> str:
    seq_m     = re.search(r"Event Sequence\s*:\s*(.+)", doc)
    event_seq = seq_m.group(1).strip() if seq_m else ""
    return (f"HDFS block anomaly: {meta.get('anomaly_types','unknown')}. "
            f"Latency: {meta.get('latency',0)}ms. "
            f"Total events: {meta.get('total_events',0)}. "
            f"Event sequence: {event_seq}")


def build_event_context(doc: str) -> str:
    seq_m = re.search(r"Event Sequence\s*:\s*(.+)", doc)
    if not seq_m:
        return "No event sequence found."
    ids = list(dict.fromkeys(re.findall(r"E\d+", seq_m.group(1))))
    return "\n".join(f"{eid}: {EVENT_DESCRIPTIONS.get(eid,'Unknown event')}" for eid in ids)


def retrieve_similar_anomalies(query: str, k: int = 3, anomaly_type_filter: str = None) -> list:
    qemb  = embed_model.encode([query]).tolist()
    where = {"primary_type": {"$eq": anomaly_type_filter}} if anomaly_type_filter else None
    res   = collection.query(query_embeddings=qemb, n_results=k, where=where)
    return [
        {"document": doc, "block_id": m.get("block_id","?"),
         "primary_type": m.get("primary_type","?"),
         "latency": m.get("latency", 0), "distance": d}
        for doc, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def compute_confidence(similar_docs: list, query_doc: str, latency: int) -> tuple:
    avg_dist = sum(d["distance"] for d in similar_docs) / len(similar_docs)
    r_score  = 0.9 if avg_dist < 0.70 else (0.7 if avg_dist < 0.80 else 0.5)
    ids      = re.findall(r"E\d+", query_doc)
    counts   = Counter(ids)
    rep_score = sum(v for v in counts.values() if v > 1) / max(len(ids), 1)
    l_score  = 0.9 if latency > 20000 else (0.7 if latency > 8000 else 0.5)
    conf     = round(min(max((r_score + rep_score + l_score) / 3, 0.0), 1.0), 2)
    label    = "high" if conf >= 0.75 else ("medium" if conf >= 0.4 else "low")
    return conf, label


def repair_json(s: str) -> dict:
    """Char-by-char JSON repair: handles trailing commas, unescaped inner quotes, truncation."""
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"(?<![\w])'([^']*)'(?![\w])", r'"\1"', s)

    result    = []
    in_string = False
    i         = 0
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
                if closes or i == len(s) - 1:
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
        if in_string:
            s += '"'
        depth = []
        for ch in s:
            if ch in ("{", "["):
                depth.append("}" if ch == "{" else "]")
            elif ch in ("}", "]") and depth:
                depth.pop()
        s += "".join(reversed(depth))
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return json.loads(s)


def generate_root_cause(query_doc: str, hist_ctx: str, query_meta: dict, similar_docs: list) -> dict:
    total_events = query_meta.get("total_events", "?")
    latency_ms   = query_meta.get("latency", "?")
    event_ref    = build_event_context(query_doc)

    user_prompt = f"""QUERY BLOCK:
{query_doc}

EVENT REFERENCE:
{event_ref}

HISTORICAL CASES:
{hist_ctx}

Return JSON in EXACTLY this field order:

{{
  "block_id": "{query_meta['block_id']}",
  "anomaly_type": "copy first type from Anomaly Type(s) in QUERY BLOCK exactly",
  "confidence": 0.85,
  "confidence_label": "high",
  "mitigation_steps": {{
    "high": ["specific urgent action 1", "specific urgent action 2"],
    "medium": ["important follow-up 1", "important follow-up 2"],
    "low": ["longer-term improvement"]
  }},
  "summary": "THIS BLOCK HAS {total_events} EVENTS AND {latency_ms}ms LATENCY. Write 1-2 sentences using these exact numbers.",
  "root_cause": "mechanism first, then cite specific event IDs and repeat counts as evidence",
  "comparison_to_historical": "exactly one sentence with one specific number — no raw dicts",
  "event_explanations": {{
    "E5": "copy from EVENT REFERENCE"
  }}
}}

MANDATORY RULES:
1. mitigation_steps.high   MUST have 2+ strings. Never [], never null.
2. mitigation_steps.medium MUST have 2+ strings. Never [], never null.
3. mitigation_steps.low    MUST have 1+ string.  Never [], never null.
4. summary MUST include {total_events} and {latency_ms}ms.
5. anomaly_type: copy first value from Anomaly Type(s) exactly.
6. comparison_to_historical: 1 sentence only, no dicts.
7. event_explanations: all unique events, copy from EVENT REFERENCE.
8. DO NOT return anything except valid JSON.
"""

    MAX_RETRIES = 3
    response    = None
    last_error  = None

    for attempt in range(MAX_RETRIES):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are an HDFS expert. Return ONLY valid JSON."},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1500,
            )
            break
        except Exception as api_err:
            last_error = api_err
            err_str    = str(api_err)
            if "413" in err_str and attempt < MAX_RETRIES - 1:
                # Compress even further and retry
                compressed = "\n\n".join([
                    f"Case {i+1}: {r['block_id']} | {r['primary_type']} | {r['latency']}ms"
                    for i, r in enumerate(similar_docs)
                ])
                user_prompt = user_prompt.replace(hist_ctx, compressed)
                continue
            if "429" in err_str and attempt < MAX_RETRIES - 1:
                time.sleep(5 * (attempt + 1))
                continue
            break

    if response is None:
        return {"error": f"Groq API failed: {str(last_error)}",
                "block_id": query_meta.get("block_id", "unknown")}

    try:
        content = response.choices[0].message.content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        try:
            parsed = repair_json(content)
        except json.JSONDecodeError as e:
            print(f"[PARSE ERROR] block={query_meta.get('block_id','?')} | {e}")
            print(f"  Raw (first 300): {content[:300]}")
            parsed = {
                "block_id": query_meta.get("block_id", "parse_error"),
                "summary": "JSON parse failed.",
                "root_cause": content[:500],
                "comparison_to_historical": "",
                "event_explanations": {},
                "anomaly_type": query_meta.get("primary_type", "unknown"),
                "confidence": 0.0, "confidence_label": "low",
                "mitigation_steps": {"high": [], "medium": [], "low": []},
                "parse_error": True,
            }

        # Enforce anomaly type
        valid_types = {"duplicate_pattern", "repetition", "missing_events", "high_latency"}
        if parsed.get("anomaly_type") not in valid_types:
            parsed["anomaly_type"] = query_meta.get("primary_type", "unknown")

        # Sanitise mitigation_steps — never allow empty tiers
        atype    = parsed.get("anomaly_type", query_meta.get("primary_type", "duplicate_pattern"))
        fallback = FALLBACK_MITIGATIONS.get(atype, FALLBACK_MITIGATIONS["duplicate_pattern"])
        mit      = parsed.get("mitigation_steps")
        if not isinstance(mit, dict):
            parsed["mitigation_steps"] = fallback
        else:
            for tier in ("high", "medium", "low"):
                cleaned = [x for x in (mit.get(tier) or []) if x and str(x).strip()]
                mit[tier] = cleaned if cleaned else fallback[tier]

        # Sanitise comparison_to_historical
        cth = parsed.get("comparison_to_historical", "")
        if isinstance(cth, (list, dict)) or (isinstance(cth, str) and cth.strip().startswith("[")):
            avg_lat = int(sum(r["latency"] for r in similar_docs) / max(len(similar_docs), 1))
            parsed["comparison_to_historical"] = (
                f"Similar pattern to retrieved cases (avg historical latency: {avg_lat}ms "
                f"vs this block: {query_meta.get('latency',0)}ms)."
            )

        # Override confidence with computed value
        conf, label = compute_confidence(similar_docs, query_doc, query_meta.get("latency", 0))
        parsed["confidence"]       = conf
        parsed["confidence_label"] = label

        return parsed

    except Exception as e:
        return {"error": f"Response processing failed: {str(e)}",
                "block_id": query_meta.get("block_id", "unknown")}


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="HDFS Anomaly Root Cause API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    query:               Optional[str] = None
    block_id:            Optional[str] = None
    anomaly_type_filter: Optional[str] = None
    k:                   Optional[int] = 3


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "blocks": "/blocks"}


@app.post("/analyze")
def analyze_log(request: AnalyzeRequest):
    if not request.query and not request.block_id:
        raise HTTPException(status_code=400, detail="Provide 'query' or 'block_id'.")

    if request.block_id:
        if request.block_id not in block_lookup:
            raise HTTPException(status_code=404,
                detail=f"block_id '{request.block_id}' not found.")
        q_doc, q_meta = block_lookup[request.block_id]
        query_str    = build_query_from_doc(q_doc, q_meta)
        atype_filter = request.anomaly_type_filter or q_meta["primary_type"]
    else:
        q_doc        = request.query
        q_meta       = {"block_id": "external_query",
                        "primary_type": request.anomaly_type_filter or "unknown",
                        "anomaly_types": request.anomaly_type_filter or "unknown",
                        "latency": 0, "total_events": 0}
        query_str    = request.query
        atype_filter = request.anomaly_type_filter

    similar = retrieve_similar_anomalies(
        query=query_str, k=request.k, anomaly_type_filter=atype_filter
    )

    hist_ctx = "\n\n".join([
        f"Case {i+1}\n" +
        compress_historical_case(r["document"], r["block_id"], r["latency"], r["distance"])
        for i, r in enumerate(similar)
        if r["block_id"] != q_meta.get("block_id", "")
    ])

    result = generate_root_cause(q_doc, hist_ctx, q_meta, similar)
    result["retrieved_block_ids"] = [r["block_id"] for r in similar]
    result["query_block_id"]      = q_meta.get("block_id", "external_query")
    return result


@app.get("/blocks")
def list_blocks(limit: int = 20, anomaly_type: Optional[str] = None):
    blocks = [
        {"block_id": m["block_id"], "primary_type": m["primary_type"], "latency_ms": m["latency"]}
        for m in all_metadata
        if not anomaly_type or m["primary_type"] == anomaly_type
    ]
    return {"total": len(blocks), "blocks": blocks[:limit]}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents_loaded": len(documents),
        "vectors_indexed":  collection.count() if collection else 0,
    }
