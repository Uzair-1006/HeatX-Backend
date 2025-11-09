# routers/ccts.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import os, json, uuid, hmac, hashlib, threading

router = APIRouter(prefix="/ccts", tags=["CCTS"])

# ---------------------------------------------------------------------
# Persistent storage (JSON file). Simple + hackathon friendly.
# Swap to Postgres later if needed.
# ---------------------------------------------------------------------
DATA_DIR = os.environ.get("CCTS_DATA_DIR", "./ccts_data")
os.makedirs(DATA_DIR, exist_ok=True)
ORG_FILE   = os.path.join(DATA_DIR, "orgs.json")
STATE_FILE = os.path.join(DATA_DIR, "state.json")

_lock = threading.Lock()

def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------
class Org(BaseModel):
    org_id: str
    name: str
    api_key: str  # demo secret; rotate in production

class TxType:
    MINT = "MINT"
    TRANSFER = "TRANSFER"
    RETIRE = "RETIRE"

class Tx(BaseModel):
    tx_id: str
    type: str = Field(pattern="^(MINT|TRANSFER|RETIRE)$")
    timestamp: str
    payload: Dict[str, Any]  # {org_id/ from/ to/ amount/ meta...}
    signature: str

class Block(BaseModel):
    index: int
    timestamp: str
    prev_hash: str
    merkle_root: str
    txs: List[Tx]
    hash: str

class RegisterOrgIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)

class MintIn(BaseModel):
    org_id: str
    amount: float = Field(gt=0)
    metadata: Optional[Dict[str, Any]] = None

class TransferIn(BaseModel):
    from_org: str
    to_org: str
    amount: float = Field(gt=0)
    metadata: Optional[Dict[str, Any]] = None

class RetireIn(BaseModel):
    org_id: str
    amount: float = Field(gt=0)
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------
# In-memory state with JSON persistence
# ---------------------------------------------------------------------
class State:
    def __init__(self):
        self.orgs: Dict[str, Org] = {}
        # balances[org_id] = float
        self.balances: Dict[str, float] = {}
        self.chain: List[Block] = []

    def to_dict(self):
        return {
            "balances": self.balances,
            "chain": [b.model_dump() for b in self.chain],
        }

    @staticmethod
    def from_dict(d: dict) -> "State":
        s = State()
        s.balances = d.get("balances", {})
        s.chain = [Block(**b) for b in d.get("chain", [])]
        return s

def _state_load() -> State:
    state = _load_json(STATE_FILE, None)
    if not state:
        s = State()
        # Genesis block
        genesis = Block(
            index=0,
            timestamp=datetime.utcnow().isoformat(),
            prev_hash="0"*64,
            merkle_root="0"*64,
            txs=[],
            hash="GENESIS"
        )
        s.chain.append(genesis)
        _state_save(s)
        return s
    return State.from_dict(state)

def _state_save(s: State):
    _save_json(STATE_FILE, s.to_dict())

def _orgs_load() -> Dict[str, Org]:
    data = _load_json(ORG_FILE, {})
    return {k: Org(**v) for k, v in data.items()}

def _orgs_save(orgs: Dict[str, Org]):
    _save_json(ORG_FILE, {k: v.model_dump() for k, v in orgs.items()})

# singletons
_ORGS = _orgs_load()
_STATE = _state_load()

# ---------------------------------------------------------------------
# Helpers (Merkle root, hashing, signing)
# ---------------------------------------------------------------------
def sha256_hex(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()

def _merkle_root(tx_ids: List[str]) -> str:
    if not tx_ids:
        return "0"*64
    level = [bytes.fromhex(sha256_hex(t.encode())) for t in tx_ids]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i+1] if i+1 < len(level) else a
            nxt.append(hashlib.sha256(a + b).digest())
        level = nxt
    return level[0].hex()

def _sign(payload: dict, api_key: str) -> str:
    msg = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(api_key.encode(), msg, hashlib.sha256).hexdigest()

def _verify_sig(payload: dict, signature: str, api_key: str) -> bool:
    expected = _sign(payload, api_key)
    return hmac.compare_digest(signature, expected)

def _require_org(org_id: str) -> Org:
    org = _ORGS.get(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org

# Dependency: demo HMAC signature header
def require_signature(x_signature: str = Header(...)) -> str:
    # returns provided signature for route handlers
    return x_signature

# ---------------------------------------------------------------------
# Blocks & transactions application
# ---------------------------------------------------------------------
def _append_block(txs: List[Tx]):
    global _STATE
    prev = _STATE.chain[-1]
    tx_ids = [t.tx_id for t in txs]
    merkle = _merkle_root(tx_ids)
    header = {
        "index": prev.index + 1,
        "timestamp": datetime.utcnow().isoformat(),
        "prev_hash": prev.hash,
        "merkle_root": merkle,
        "tx_ids": tx_ids
    }
    block_hash = sha256_hex(json.dumps(header, sort_keys=True).encode())
    block = Block(
        index=header["index"],
        timestamp=header["timestamp"],
        prev_hash=header["prev_hash"],
        merkle_root=merkle,
        txs=txs,
        hash=block_hash
    )
    _STATE.chain.append(block)

def _apply_tx(tx: Tx):
    t = tx.type
    p = tx.payload
    if t == TxType.MINT:
        org_id = p["org_id"]
        amt = float(p["amount"])
        _STATE.balances[org_id] = _STATE.balances.get(org_id, 0.0) + amt
    elif t == TxType.TRANSFER:
        src, dst, amt = p["from_org"], p["to_org"], float(p["amount"])
        if _STATE.balances.get(src, 0.0) < amt:
            raise HTTPException(status_code=400, detail="Insufficient balance")
        _STATE.balances[src] = _STATE.balances.get(src, 0.0) - amt
        _STATE.balances[dst] = _STATE.balances.get(dst, 0.0) + amt
    elif t == TxType.RETIRE:
        org_id = p["org_id"]
        amt = float(p["amount"])
        if _STATE.balances.get(org_id, 0.0) < amt:
            raise HTTPException(status_code=400, detail="Insufficient balance")
        _STATE.balances[org_id] = _STATE.balances.get(org_id, 0.0) - amt
    else:
        raise HTTPException(status_code=400, detail="Unknown tx type")

def _create_and_commit_tx(t_type: str, payload: dict, signature: str):
    tx = Tx(
        tx_id=str(uuid.uuid4()),
        type=t_type,
        timestamp=datetime.utcnow().isoformat(),
        payload=payload,
        signature=signature
    )
    # apply & commit in a single-TX block (simple)
    _apply_tx(tx)
    _append_block([tx])

# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@router.post("/orgs/register")
def register_org(body: RegisterOrgIn):
    with _lock:
        org_id = str(uuid.uuid4())
        api_key = uuid.uuid4().hex  # demo secret; store hashed in prod
        org = Org(org_id=org_id, name=body.name, api_key=api_key)
        _ORGS[org_id] = org
        _orgs_save(_ORGS)
        if org_id not in _STATE.balances:
            _STATE.balances[org_id] = 0.0
            _state_save(_STATE)
        return {"org_id": org_id, "api_key": api_key}

@router.get("/orgs")
def list_orgs():
    return [{"org_id": o.org_id, "name": o.name} for o in _ORGS.values()]

@router.get("/balance/{org_id}")
def balance(org_id: str):
    _require_org(org_id)
    return {"org_id": org_id, "balance": _STATE.balances.get(org_id, 0.0)}

@router.get("/ledger")
def ledger():
    return {
        "height": len(_STATE.chain) - 1,
        "balances": _STATE.balances,
        "blocks": [b.model_dump() for b in _STATE.chain[-50:]]  # tail
    }

@router.post("/mint")
def mint(body: MintIn, x_signature: str = Depends(require_signature)):
    with _lock:
        org = _require_org(body.org_id)
        payload = {
            "amount": body.amount,
            "metadata": body.metadata or {},
            "org_id": body.org_id
        }

        # 🧠 DEBUG LOGGING (add this)
        import json, hashlib, hmac
        server_msg = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        expected_sig = hmac.new(org.api_key.encode(), server_msg, hashlib.sha256).hexdigest()
        print("\n---- DEBUG SIGNATURE VERIFICATION ----")
        print("Server JSON:", server_msg.decode())
        print("Expected signature:", expected_sig)
        print("Received signature:", x_signature)
        print("-------------------------------------\n")

        if not _verify_sig(payload, x_signature, org.api_key):
            raise HTTPException(status_code=401, detail="Invalid signature")

        _create_and_commit_tx(TxType.MINT, payload, x_signature)
        _state_save(_STATE)
        return {"status": "ok", "org_id": body.org_id, "delta": +body.amount}

@router.post("/transfer")
def transfer(body: TransferIn, x_signature: str = Depends(require_signature)):
    with _lock:
        src = _require_org(body.from_org)
        _require_org(body.to_org)
        payload = {
            "amount": float(body.amount),
            "from_org": body.from_org,
            "metadata": body.metadata or {},
            "to_org": body.to_org
        }

        # 🧠 DEBUG
        import json, hashlib, hmac
        server_msg = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        expected_sig = hmac.new(src.api_key.encode(), server_msg, hashlib.sha256).hexdigest()
        print("\n---- DEBUG SIGNATURE VERIFICATION (TRANSFER) ----")
        print("Server JSON:", server_msg.decode())
        print("Expected signature:", expected_sig)
        print("Received signature:", x_signature)
        print("-----------------------------------------------\n")

        if not _verify_sig(payload, x_signature, src.api_key):
            raise HTTPException(status_code=401, detail="Invalid signature")

        _create_and_commit_tx(TxType.TRANSFER, payload, x_signature)
        _state_save(_STATE)
        return {"status": "ok", "from": body.from_org, "to": body.to_org, "delta": -body.amount}

@router.post("/retire")
def retire(body: RetireIn, x_signature: str = Depends(require_signature)):
    with _lock:
        org = _require_org(body.org_id)
        payload = {
            "org_id": body.org_id,
            "amount": body.amount,
            "reason": body.reason,
            "metadata": body.metadata
        }
        if not _verify_sig(payload, x_signature, org.api_key):
            raise HTTPException(status_code=401, detail="Invalid signature")
        _create_and_commit_tx(TxType.RETIRE, payload, x_signature)
        _state_save(_STATE)
        return {"status": "ok", "org_id": body.org_id, "retired": body.amount}

@router.get("/tx/{tx_id}")
def get_tx(tx_id: str):
    for b in _STATE.chain:
        for t in b.txs:
            if t.tx_id == tx_id:
                return {"block_index": b.index, "tx": t.model_dump()}
    raise HTTPException(status_code=404, detail="Transaction not found")

@router.get("/audit/verify")
def audit_verify():
    # Verify chain linkage + merkle roots
    chain = _STATE.chain
    for i, b in enumerate(chain):
        if i == 0:  # genesis
            if b.hash != "GENESIS":
                return {"ok": False, "error": "Genesis block corrupted"}
            continue
        prev = chain[i-1]
        if b.prev_hash != prev.hash:
            return {"ok": False, "error": f"Broken link at block {i}"}
        recomputed_merkle = _merkle_root([t.tx_id for t in b.txs])
        if recomputed_merkle != b.merkle_root:
            return {"ok": False, "error": f"Merkle mismatch at block {i}"}
        header = {
            "index": b.index,
            "timestamp": b.timestamp,
            "prev_hash": b.prev_hash,
            "merkle_root": b.merkle_root,
            "tx_ids": [t.tx_id for t in b.txs]
        }
        if sha256_hex(json.dumps(header, sort_keys=True).encode()) != b.hash:
            return {"ok": False, "error": f"Hash mismatch at block {i}"}
    return {"ok": True, "height": len(chain) - 1}
