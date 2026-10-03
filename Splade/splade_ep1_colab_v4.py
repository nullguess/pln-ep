"""
EP1 - ACH2118 Introdução ao Processamento de Língua Natural
Abordagem: SPLADE-PT-BR como representação lexical esparsa aprendida
           + classificadores lineares do scikit-learn.

Este arquivo NÃO faz fine-tuning do SPLADE/BERTimbau.
O SPLADE-PT-BR é usado apenas como codificador congelado de texto.

Dependências fixadas em requirements_splade_ep1_v4.txt.
Instale as versões e reinicie o kernel antes de executar o notebook.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

# Versões verificadas no ambiente de teste isolado do servidor.
CODE_VERSION = "SPLADE_EP1_COLAB_V4"
REQUIRED_VERSIONS = {'torch': '2.11.0', 'transformers': '4.57.3', 'accelerate': '1.15.0', 'huggingface_hub': '0.36.2', 'tokenizers': '0.22.2', 'safetensors': '0.8.0', 'numpy': '2.3.5', 'pandas': '2.2.3', 'scipy': '1.17.0', 'scikit-learn': '1.8.0', 'tqdm': '4.70.1', 'psutil': '7.2.2', 'openpyxl': '3.1.5', 'setuptools': '81.0.0'}

def validate_environment() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    module_names = {"scikit-learn": "sklearn", "huggingface_hub": "huggingface_hub"}
    found = {}
    errors = []
    for package, expected in REQUIRED_VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = "ausente"
        found[package] = actual
        comparable = actual.split("+")[0] if package == "torch" else actual
        if comparable != expected:
            errors.append(f"{package}: instalado={actual}; esperado={expected}")
        module = sys.modules.get(module_names.get(package, package))
        loaded = getattr(module, "__version__", None) if module is not None else None
        if loaded is not None:
            loaded_comparable = loaded.split("+")[0] if package == "torch" else loaded
            if loaded_comparable != expected:
                errors.append(f"{package}: versão ainda carregada no kernel={loaded}")
    if errors:
        raise RuntimeError(
            "Ambiente diferente do verificado. Execute a célula 1 de instalação "
            "(ou pip install -r requirements_splade_ep1_v4.txt), reinicie o kernel "
            "e execute desde o início.\n" + "\n".join(errors)
        )
    return found

validate_environment()
# Esta solução usa exclusivamente o backend PyTorch.
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

import torch
import transformers
from huggingface_hub import hf_hub_download
from accelerate import init_empty_weights
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig, PreTrainedModel

try:
    import psutil
except ImportError:  # memória RSS vira opcional
    psutil = None


# -----------------------------------------------------------------------------
# Configuração
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 10
LABELS = ["c1", "c234", "c5"]

TRAIN_PATH = Path("dados/train.xlsx")
TEST_PATH = Path("dados/test1.xlsx")
CACHE_DIR = Path("cache_splade")
RESULTS_DIR = Path("resultados_splade")

MODEL_ID = "AxelPCG/splade-pt-br"
# Snapshot conhecido/reprodutível que contém modeling_splade.py e os pesos.
MODEL_REVISION = "fff5e6a3a3e0dc834227d4600a90a44025863bb6"
ENCODER_CACHE_VERSION = "v4_direct_checkpoint_torch211"
MAX_LENGTH = 256
BATCH_SIZE = 4
USE_CACHE = True

# Comparações recuperadas dos experimentos anteriores do projeto.
# Apenas o baseline abaixo usa o mesmo protocolo de 10 folds e é comparação direta.
PREVIOUS_RESULTS = [
    {
        "modelo": "Baseline TF-IDF + Regressão Logística",
        "acuracia_media": 0.4565,
        "folds": 10,
        "comparacao_direta": True,
        "observacao": "baseline anterior do projeto; treino médio ~69,52%",
    },
    {
        "modelo": "SetFit + CosineSimilarityLoss + LR balanced",
        "acuracia_media": 0.4012,
        "folds": 3,
        "comparacao_direta": False,
        "observacao": "protocolo de 3 folds; usar apenas como referência contextual",
    },
]


# -----------------------------------------------------------------------------
# Utilitários de logging/memória
# -----------------------------------------------------------------------------
def rss_gb() -> float | None:
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


def log(message: str) -> None:
    memoria = rss_gb()
    sufixo = f" | RSS≈{memoria:.2f} GB" if memoria is not None else ""
    print(f"[{time.strftime('%H:%M:%S')}] {message}{sufixo}", flush=True)


def sparse_memory_mb(matrix: sparse.spmatrix) -> float:
    csr = matrix.tocsr(copy=False)
    return (csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes) / (1024**2)


def log_sparse_matrix(name: str, matrix: sparse.spmatrix) -> None:
    total = matrix.shape[0] * matrix.shape[1]
    sparsity = 1.0 - (matrix.nnz / total) if total else 1.0
    dense_mb = total * np.dtype(np.float32).itemsize / (1024**2)
    log(
        f"{name}: shape={matrix.shape}, nnz={matrix.nnz:,}, "
        f"esparsidade={sparsity:.4%}, CSR≈{sparse_memory_mb(matrix):.2f} MB, "
        f"denso float32≈{dense_mb:.2f} MB"
    )


def log_versions() -> None:
    print(f"Código ativo: {CODE_VERSION}")
    print(f"Python: {platform.python_version()}")
    for package, installed in validate_environment().items():
        print(f"  {package:<16}: {installed} [versão verificada]")
    print(f"Dispositivo: {'cuda' if torch.cuda.is_available() else 'cpu'}")


# -----------------------------------------------------------------------------
# Dados
# -----------------------------------------------------------------------------
def load_datasets(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray, list[str]]:
    log(f"Carregando treino: {train_path}")
    train_df = pd.read_excel(train_path)
    log(f"Carregando teste: {test_path}")
    test_df = pd.read_excel(test_path)

    required_train = {"resp_text", "clarity"}
    if not required_train.issubset(train_df.columns):
        raise ValueError(f"Treino deve conter as colunas {sorted(required_train)}")
    if "resp_text" not in test_df.columns:
        raise ValueError("Teste deve conter a coluna 'resp_text'")

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["resp_text"] = train_df["resp_text"].fillna("").astype(str)
    test_df["resp_text"] = test_df["resp_text"].fillna("").astype(str)
    train_df["clarity"] = train_df["clarity"].astype(str)

    y = train_df["clarity"].to_numpy()
    observed = set(np.unique(y))
    expected = set(LABELS)
    if observed != expected:
        raise ValueError(f"Rótulos encontrados={sorted(observed)}; esperados={LABELS}")

    x_train_text = train_df["resp_text"].tolist()
    x_test_text = test_df["resp_text"].tolist()

    log(f"Treino carregado: {len(train_df):,} exemplos")
    log(f"Teste carregado: {len(test_df):,} exemplos")
    print("Distribuição das classes no treino:")
    print(train_df["clarity"].value_counts().reindex(LABELS))

    return train_df, test_df, x_train_text, y, x_test_text


# -----------------------------------------------------------------------------
# SPLADE-PT-BR congelado -> scipy.sparse.csr_matrix float32
# -----------------------------------------------------------------------------
def import_splade_class(model_id: str = MODEL_ID, revision: str = MODEL_REVISION):
    """
    Baixa apenas o arquivo de definição do próprio repositório do modelo e importa
    a classe Splade diretamente. Evita depender de um checkout separado do pacote
    SPLADE e mantém a arquitetura usada pelo checkpoint.
    """
    log(f"Obtendo definição do SPLADE em {model_id}@{revision[:8]}")
    modeling_path = hf_hub_download(
        repo_id=model_id,
        filename="modeling_splade.py",
        revision=revision,
    )
    spec = importlib.util.spec_from_file_location("splade_ptbr_modeling", modeling_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Não foi possível importar modeling_splade.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Splade


def load_splade_encoder(
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Splade = import_splade_class(model_id, revision)

    # O construtor original chama AutoModelForMaskedLM.from_pretrained().
    # Quando o carregamento externo cria o SPLADE em meta, essa segunda chamada
    # falha. Aqui só criamos a arquitetura e depois atribuimos o checkpoint.
    class SpladeFromConfig(Splade):
        def __init__(self, config):
            PreTrainedModel.__init__(self, config)
            self.transformer = AutoModelForMaskedLM.from_config(config)
            self.aggregation = getattr(config, "aggregation", "max")
            self.fp16 = getattr(config, "fp16", True)

    torch_version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if torch_version < (2, 6):
        raise RuntimeError("Este carregador requer PyTorch >= 2.6. Atualize e reinicie o kernel.")

    config_path = hf_hub_download(
        repo_id=model_id, filename="config.json", revision=revision
    )
    with open(config_path, encoding="utf-8") as file:
        config_dict = json.load(file)
    if config_dict.get("model_type") != "bert":
        raise ValueError("Este carregador espera a configuração BERT da revisão fixa do SPLADE-PT-BR.")
    config = BertConfig.from_dict(config_dict)

    log(f"Carregando tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    log("Criando arquitetura SPLADE sem alocar uma cópia inicial dos pesos")
    # Apenas parâmetros ficam em meta; buffers como position_ids ficam na CPU.
    # Nenhum from_pretrained é chamado dentro deste contexto.
    with torch.device("cpu"), init_empty_weights(include_buffers=False):
        model = SpladeFromConfig(config)

    log(f"Carregando checkpoint SPLADE com mmap: {model_id}@{revision[:8]}")
    weights_path = hf_hub_download(
        repo_id=model_id, filename="pytorch_model.bin", revision=revision
    )
    state_dict = torch.load(
        weights_path, map_location="cpu", weights_only=True, mmap=True
    )
    # Suporta tanto o checkpoint BERT MLM quanto o wrapper oficial SPLADE.
    # strict=True impede usar pesos aleatórios ou ignorar camadas incompatíveis.
    if state_dict and all(key.startswith("transformer.") for key in state_dict):
        model.load_state_dict(state_dict, strict=True, assign=True)
        log("Pesos do wrapper SPLADE carregados: todas as chaves conferidas")
    elif state_dict and all(key.startswith(("bert.", "cls.")) for key in state_dict):
        model.transformer.load_state_dict(state_dict, strict=True, assign=True)
        log("Pesos BERT MLM do checkpoint SPLADE carregados: todas as chaves conferidas")
    else:
        raise RuntimeError("Formato de chaves inesperado no checkpoint SPLADE; carregamento interrompido.")
    del state_dict
    model.transformer.tie_weights()

    meta_tensors = [
        name for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor.is_meta
    ]
    if meta_tensors:
        raise RuntimeError(f"Checkpoint incompleto: tensores ainda em meta: {meta_tensors[:5]}")
    model.eval()
    model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    log(f"SPLADE pronto em {device}. Fine-tuning desativado.")
    return tokenizer, model, device


def _fingerprint_texts(texts: list[str], prefix: str) -> str:
    digest = hashlib.sha256()
    digest.update(MODEL_ID.encode("utf-8"))
    digest.update(MODEL_REVISION.encode("utf-8"))
    digest.update(ENCODER_CACHE_VERSION.encode("ascii"))
    digest.update(str(MAX_LENGTH).encode("ascii"))
    digest.update(prefix.encode("utf-8"))
    for text in texts:
        digest.update(text.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def cache_path_for(texts: list[str], prefix: str) -> Path:
    return CACHE_DIR / f"{prefix}_{_fingerprint_texts(texts, prefix)}.npz"


def encode_splade_sparse(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    *,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    description: str = "SPLADE",
) -> sparse.csr_matrix:
    """
    Codifica lotes pequenos. Apenas o lote corrente existe como tensor denso;
    o conjunto completo é montado em CSR float32.
    """
    blocks: list[sparse.csr_matrix] = []
    use_amp = device.type == "cuda"
    amp_context: Callable[[], object]
    amp_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else nullcontext()
    )

    model.eval()
    starts = range(0, len(texts), batch_size)
    progress = tqdm(starts, total=(len(texts) + batch_size - 1) // batch_size, desc=description)

    with torch.inference_mode():
        for batch_number, start in enumerate(progress, start=1):
            batch_texts = texts[start : start + batch_size]
            tokens = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {name: tensor.to(device, non_blocking=True) for name, tensor in tokens.items()}

            with amp_context():
                # As respostas são tratadas como documentos, portanto usamos d_rep.
                dense_rep = model(d_kwargs=tokens)["d_rep"]

            batch_np = dense_rep.detach().float().cpu().numpy().astype(np.float32, copy=False)
            batch_csr = sparse.csr_matrix(batch_np, dtype=np.float32)
            batch_csr.eliminate_zeros()
            blocks.append(batch_csr)

            del tokens, dense_rep, batch_np, batch_csr
            if device.type == "cuda" and batch_number % 25 == 0:
                torch.cuda.empty_cache()

    if not blocks:
        return sparse.csr_matrix((0, 0), dtype=np.float32)

    matrix = sparse.vstack(blocks, format="csr", dtype=np.float32)
    matrix.sort_indices()
    return matrix


def get_or_create_representation(
    texts: list[str],
    prefix: str,
    tokenizer=None,
    model=None,
    device: torch.device | None = None,
) -> sparse.csr_matrix:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path_for(texts, prefix)

    if USE_CACHE and cache_path.exists():
        log(f"Carregando representação em cache: {cache_path}")
        matrix = sparse.load_npz(cache_path).tocsr().astype(np.float32, copy=False)
        log_sparse_matrix(prefix, matrix)
        return matrix

    if tokenizer is None or model is None or device is None:
        raise ValueError("Tokenizer/model/device são necessários quando não há cache")

    log(f"Gerando representação SPLADE para '{prefix}'")
    matrix = encode_splade_sparse(
        texts,
        tokenizer,
        model,
        device,
        description=f"SPLADE {prefix}",
    )
    log_sparse_matrix(prefix, matrix)

    if USE_CACHE:
        sparse.save_npz(cache_path, matrix, compressed=True)
        log(f"Cache salvo: {cache_path}")

    return matrix


def build_representations(
    x_train_text: list[str],
    x_test_text: list[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    train_cache = cache_path_for(x_train_text, "train")
    test_cache = cache_path_for(x_test_text, "test")

    tokenizer = model = device = None
    if not (USE_CACHE and train_cache.exists() and test_cache.exists()):
        tokenizer, model, device = load_splade_encoder()

    try:
        x_train = get_or_create_representation(
            x_train_text, "train", tokenizer, model, device
        )
        x_test = get_or_create_representation(
            x_test_text, "test", tokenizer, model, device
        )
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log("Codificador SPLADE liberado da memória")

    if x_train.shape[1] != x_test.shape[1]:
        raise RuntimeError("Treino e teste ficaram com números diferentes de características")

    return x_train, x_test


# -----------------------------------------------------------------------------
# Classificadores sklearn adequados à representação esparsa
# -----------------------------------------------------------------------------
def classifier_factories() -> dict[str, Callable[[], object]]:
    return {
        # Principal: margem máxima em espaço esparso de alta dimensão.
        "LinearSVC": lambda: LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual=True,
            max_iter=10_000,
            random_state=RANDOM_STATE,
        ),
        # Comparação linear adicional: mantém a mesma representação e muda apenas
        # a função de decisão; SAGA trabalha diretamente com entrada esparsa.
        "LogisticRegression": lambda: LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="saga",
            max_iter=1_000,
            tol=1e-3,
            random_state=RANDOM_STATE,
        ),
    }


def _scores_for_oof(clf, x_val: sparse.csr_matrix) -> tuple[np.ndarray, str]:
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(x_val)
        return np.asarray(scores, dtype=np.float32), "decision_function"
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(x_val)
        return np.asarray(scores, dtype=np.float32), "predict_proba"
    raise TypeError("Classificador não fornece decision_function nem predict_proba")


def evaluate_classifier_cv(
    model_name: str,
    factory: Callable[[], object],
    x: sparse.csr_matrix,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    n = len(y)
    oof_pred = np.empty(n, dtype=object)
    oof_fold = np.zeros(n, dtype=np.int16)
    oof_scores = np.full((n, len(LABELS)), np.nan, dtype=np.float32)
    fold_rows: list[dict] = []
    score_type = None

    log(f"Iniciando CV de {model_name}: {len(splits)} folds")

    for fold, (train_idx, val_idx) in enumerate(
        tqdm(splits, desc=f"CV {model_name}"), start=1
    ):
        fold_start = time.time()
        clf = factory()

        x_train_fold = x[train_idx]
        x_val_fold = x[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        log(
            f"{model_name} | fold {fold:02d}/{len(splits)} | "
            f"fit={len(train_idx):,} | val={len(val_idx):,}"
        )
        clf.fit(x_train_fold, y_train_fold)

        train_pred = clf.predict(x_train_fold)
        val_pred = clf.predict(x_val_fold)
        val_scores, current_score_type = _scores_for_oof(clf, x_val_fold)
        score_type = current_score_type

        # Alinha as colunas dos scores à ordem fixa c1,c234,c5.
        if val_scores.ndim == 1:
            raise RuntimeError("Esperava scores multiclasse com 3 colunas")
        class_to_col = {label: idx for idx, label in enumerate(clf.classes_)}
        for target_col, label in enumerate(LABELS):
            oof_scores[val_idx, target_col] = val_scores[:, class_to_col[label]]

        oof_pred[val_idx] = val_pred
        oof_fold[val_idx] = fold

        train_acc = accuracy_score(y_train_fold, train_pred)
        val_acc = accuracy_score(y_val_fold, val_pred)
        macro_f1 = f1_score(y_val_fold, val_pred, labels=LABELS, average="macro")
        elapsed = time.time() - fold_start

        fold_rows.append(
            {
                "modelo": model_name,
                "fold": fold,
                "n_treino": len(train_idx),
                "n_validacao": len(val_idx),
                "acuracia_treino": train_acc,
                "acuracia_validacao": val_acc,
                "gap_treino_validacao": train_acc - val_acc,
                "f1_macro_validacao": macro_f1,
                "tempo_s": elapsed,
            }
        )
        log(
            f"{model_name} | fold {fold:02d} | "
            f"acc treino={train_acc:.2%} | acc val={val_acc:.2%} | "
            f"F1 macro={macro_f1:.2%} | tempo={elapsed:.1f}s"
        )

        del clf, x_train_fold, x_val_fold
        gc.collect()

    fold_df = pd.DataFrame(fold_rows)
    mean_acc = float(fold_df["acuracia_validacao"].mean())
    std_acc = float(fold_df["acuracia_validacao"].std(ddof=1))
    mean_train = float(fold_df["acuracia_treino"].mean())
    mean_gap = float(fold_df["gap_treino_validacao"].mean())

    report_dict = classification_report(
        y,
        oof_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    cm = confusion_matrix(y, oof_pred, labels=LABELS)

    log(
        f"{model_name} concluído | acc={mean_acc:.2%} ± {std_acc:.2%} | "
        f"treino={mean_train:.2%} | gap={mean_gap:.2%}"
    )

    return {
        "model_name": model_name,
        "folds": fold_df,
        "oof_pred": oof_pred,
        "oof_fold": oof_fold,
        "oof_scores": oof_scores,
        "score_type": score_type,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_train_accuracy": mean_train,
        "mean_gap": mean_gap,
        "report": report_df,
        "confusion_matrix": cm,
    }


def save_cv_artifacts(result: dict, y_true: np.ndarray) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = result["model_name"].lower().replace(" ", "_")

    result["folds"].to_csv(RESULTS_DIR / f"folds_{slug}.csv", index=False)
    result["report"].to_csv(RESULTS_DIR / f"metricas_classes_{slug}.csv")

    cm_df = pd.DataFrame(result["confusion_matrix"], index=LABELS, columns=LABELS)
    cm_df.to_csv(RESULTS_DIR / f"matriz_confusao_{slug}.csv")

    oof_df = pd.DataFrame(
        {
            "row_id": np.arange(len(y_true)),
            "fold": result["oof_fold"],
            "clarity_real": y_true,
            "clarity_pred": result["oof_pred"],
            "score_c1": result["oof_scores"][:, 0],
            "score_c234": result["oof_scores"][:, 1],
            "score_c5": result["oof_scores"][:, 2],
        }
    )
    oof_df.to_csv(RESULTS_DIR / f"oof_{slug}.csv", index=False)
    log(f"Artefatos OOF/metrics salvos para {result['model_name']}")


def run_cross_validation(
    x_train: sparse.csr_matrix,
    y: np.ndarray,
) -> dict[str, dict]:
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    # Os mesmos índices são reutilizados por todos os classificadores.
    splits = list(cv.split(np.zeros(len(y), dtype=np.int8), y))

    results: dict[str, dict] = {}
    for model_name, factory in classifier_factories().items():
        result = evaluate_classifier_cv(model_name, factory, x_train, y, splits)
        save_cv_artifacts(result, y)
        results[model_name] = result
    return results


# -----------------------------------------------------------------------------
# Relatórios, análise curta e comparação com experimentos anteriores
# -----------------------------------------------------------------------------
def model_summary(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        rows.append(
            {
                "modelo": name,
                "acuracia_media": result["mean_accuracy"],
                "desvio_acuracia": result["std_accuracy"],
                "acuracia_treino_media": result["mean_train_accuracy"],
                "gap_medio": result["mean_gap"],
                "f1_macro_oof": float(result["report"].loc["macro avg", "f1-score"]),
            }
        )
    return pd.DataFrame(rows).sort_values("acuracia_media", ascending=False).reset_index(drop=True)


def normalized_confusion(cm: np.ndarray) -> pd.DataFrame:
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0,
    )
    return pd.DataFrame(normalized, index=LABELS, columns=LABELS)


def short_behavior_analysis(result: dict) -> str:
    cm = result["confusion_matrix"]
    row_sums = cm.sum(axis=1)
    recalls = np.divide(
        np.diag(cm),
        row_sums,
        out=np.zeros(len(LABELS), dtype=float),
        where=row_sums != 0,
    )

    best_idx = int(np.argmax(recalls))
    worst_idx = int(np.argmin(recalls))

    offdiag = cm.copy()
    np.fill_diagonal(offdiag, 0)
    src_idx, dst_idx = np.unravel_index(np.argmax(offdiag), offdiag.shape)
    largest_error = int(offdiag[src_idx, dst_idx])
    largest_error_rate = (
        largest_error / row_sums[src_idx] if row_sums[src_idx] else 0.0
    )

    c234_idx = LABELS.index("c234")
    c234_total = row_sums[c234_idx]
    c234_to_c1 = cm[c234_idx, LABELS.index("c1")]
    c234_to_c5 = cm[c234_idx, LABELS.index("c5")]

    lines = [
        f"Classe com maior recall: {LABELS[best_idx]} ({recalls[best_idx]:.2%}).",
        f"Classe com menor recall: {LABELS[worst_idx]} ({recalls[worst_idx]:.2%}).",
        (
            f"Confusão mais frequente: {LABELS[src_idx]} → {LABELS[dst_idx]}: "
            f"{largest_error} exemplos ({largest_error_rate:.2%} da classe de origem)."
        ),
        (
            f"c234: {int(c234_to_c1)} erros para c1 e {int(c234_to_c5)} erros para c5 "
            f"em {int(c234_total)} exemplos."
        ),
        (
            f"Gap médio treino-validação: {result['mean_gap']:.2%}. "
            f"Desvio da acurácia entre folds: {result['std_accuracy']:.2%}."
        ),
    ]
    return "\n".join(lines)


def compare_with_previous(summary_df: pd.DataFrame) -> pd.DataFrame:
    current = summary_df[["modelo", "acuracia_media"]].copy()
    current["folds"] = N_SPLITS
    current["comparacao_direta"] = True
    current["observacao"] = "SPLADE-PT-BR congelado + sklearn"

    previous = pd.DataFrame(PREVIOUS_RESULTS)
    comparison = pd.concat([current, previous], ignore_index=True)

    baseline = next(
        item for item in PREVIOUS_RESULTS if item["comparacao_direta"]
    )["acuracia_media"]
    comparison["delta_vs_baseline_10fold_pp"] = (
        comparison["acuracia_media"] - baseline
    ) * 100
    return comparison


def approach_conclusion(summary_df: pd.DataFrame) -> str:
    best = summary_df.iloc[0]
    baseline = next(
        item for item in PREVIOUS_RESULTS if item["comparacao_direta"]
    )["acuracia_media"]
    delta = float(best["acuracia_media"] - baseline)

    if delta <= 0:
        return (
            f"Conclusão: não vale manter esta abordagem como linha principal neste estado. "
            f"O melhor classificador ({best['modelo']}) ficou {abs(delta):.2%} abaixo do "
            f"baseline TF-IDF + Regressão Logística de 10 folds."
        )
    if delta < 0.01:
        return (
            f"Conclusão: houve ganho de apenas {delta:.2%} sobre o baseline de 10 folds. "
            "O resultado é pequeno; a abordagem ainda não deve ser tratada como melhoria "
            "confirmada sem evidência adicional."
        )
    return (
        f"Conclusão: a abordagem merece ser mantida como candidata. O melhor classificador "
        f"({best['modelo']}) superou o baseline de 10 folds em {delta:.2%}. "
        "A decisão final ainda deve considerar estabilidade entre folds e erros por classe."
    )


def print_detailed_results(results: dict[str, dict]) -> pd.DataFrame:
    for name, result in results.items():
        print("\n" + "=" * 90)
        print(name)
        print("=" * 90)
        print("\nResultados dos 10 folds:")
        print(
            result["folds"][
                [
                    "fold",
                    "acuracia_treino",
                    "acuracia_validacao",
                    "gap_treino_validacao",
                    "f1_macro_validacao",
                    "tempo_s",
                ]
            ].to_string(index=False)
        )
        print(
            f"\nAcurácia média: {result['mean_accuracy']:.2%} "
            f"± {result['std_accuracy']:.2%}"
        )
        print(f"Acurácia média de treino: {result['mean_train_accuracy']:.2%}")
        print(f"Gap médio treino-validação: {result['mean_gap']:.2%}")

        print("\nMétricas por classe (OOF agregado):")
        rows = LABELS + ["macro avg", "weighted avg"]
        print(result["report"].loc[rows].to_string())

        cm_df = pd.DataFrame(result["confusion_matrix"], index=LABELS, columns=LABELS)
        print("\nMatriz de confusão agregada (linhas=reais, colunas=preditas):")
        print(cm_df.to_string())
        print("\nMatriz de confusão normalizada por classe real:")
        print(normalized_confusion(result["confusion_matrix"]).to_string(float_format=lambda x: f"{x:.2%}"))

        print("\nAnálise curta:")
        print(short_behavior_analysis(result))

    summary = model_summary(results)
    print("\n" + "=" * 90)
    print("Resumo dos classificadores")
    print("=" * 90)
    print(summary.to_string(index=False))

    comparison = compare_with_previous(summary)
    print("\nComparação com resultados anteriores:")
    print(comparison.to_string(index=False))

    print("\n" + approach_conclusion(summary))
    return summary


# -----------------------------------------------------------------------------
# Treino final e rotulação do conjunto de teste
# -----------------------------------------------------------------------------
def train_final_and_predict(
    summary_df: pd.DataFrame,
    x_train: sparse.csr_matrix,
    y: np.ndarray,
    x_test: sparse.csr_matrix,
    test_df: pd.DataFrame,
) -> tuple[object, pd.DataFrame]:
    best_name = str(summary_df.iloc[0]["modelo"])
    factory = classifier_factories()[best_name]
    clf = factory()

    log(f"Treinando modelo final em 100% do treino: {best_name}")
    clf.fit(x_train, y)
    test_pred = clf.predict(x_test)

    if not set(np.unique(test_pred)).issubset(set(LABELS)):
        raise RuntimeError("Modelo gerou rótulo fora de c1/c234/c5")

    output = pd.DataFrame(
        {
            "resp_text": test_df["resp_text"].to_numpy(),
            "clarity": test_pred,
        }
    )
    if len(output) != len(test_df):
        raise RuntimeError("Número de linhas do arquivo rotulado difere do teste")
    if list(output.columns) != ["resp_text", "clarity"]:
        raise RuntimeError("Formato do arquivo final foi alterado")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "teste_rotulado_splade.xlsx"
    output.to_excel(output_path, index=False)
    log(f"Teste rotulado salvo em: {output_path}")
    return clf, output


def save_summary_json(summary_df: pd.DataFrame, results: dict[str, dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "representation": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "max_length": MAX_LENGTH,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "models": {},
    }
    for name, result in results.items():
        payload["models"][name] = {
            "mean_accuracy": result["mean_accuracy"],
            "std_accuracy": result["std_accuracy"],
            "mean_train_accuracy": result["mean_train_accuracy"],
            "mean_gap": result["mean_gap"],
            "macro_f1_oof": float(result["report"].loc["macro avg", "f1-score"]),
        }
    payload["selected_classifier"] = str(summary_df.iloc[0]["modelo"])

    with open(RESULTS_DIR / "resumo_experimento.json", "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Execução completa pelo arquivo .py
# -----------------------------------------------------------------------------
def main() -> None:
    log_versions()
    train_df, test_df, x_train_text, y, x_test_text = load_datasets()
    x_train, x_test = build_representations(x_train_text, x_test_text)

    results = run_cross_validation(x_train, y)
    summary = print_detailed_results(results)
    save_summary_json(summary, results)

    # A rotulação é gerada mesmo que a conclusão seja negativa, para permitir
    # inspeção; não implica que o modelo deva ser escolhido para a entrega final.
    train_final_and_predict(summary, x_train, y, x_test, test_df)


if __name__ == "__main__":
    main()
