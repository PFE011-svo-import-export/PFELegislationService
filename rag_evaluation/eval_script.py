import json
import os
import time

from app.core.config import settings
from app.core.dependencies import (
    get_anthropic_client,
    get_cohere_client,
    get_openai_client,
    get_vector_store,
)
from app.services.rag_service import RagService
from app.services.chat_service import ChatService

# les questions + reponses attendues, on lit seulement la dedans
DATASET_PATH = "eval_questions.json"

# ce qu'on genere, c'est ce fichier que ragas va manger
RESULTS_PATH = "ragas_dataset.json"

# tant que answer == ça, l'item est pas encore fait
PLACEHOLDER = "..."

# None = tout, sinon un nombre pour tester vite fait sur les premieres
LIMIT = None
# on sauvegarde tous les N items (1 = à chaque fois)
SAVE_EVERY = 1


def build_chat_service() -> ChatService:
    """Remonte le ChatService à la main vu qu'on est hors FastAPI (pas de Depends ici)."""
    rag_service = RagService(
        client=get_openai_client(),
        embed_model=settings.openai_api_model,
        vector_store=get_vector_store(),
        reranker_model=get_cohere_client(),
    )
    return ChatService(client=get_anthropic_client(), rag_service=rag_service)


def save_json(path: str, payload) -> None:
    # on passe par un .tmp puis rename, comme ça si ça plante en cours
    # de route on se retrouve pas avec un json à moitié écrit
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def is_generated(item: dict) -> bool:
    # vrai réponse = ok, le "..." du fichier source compte pas
    answer = item.get("answer")
    return bool(answer) and answer != PLACEHOLDER


def build_results(dataset: dict) -> list:
    # format ragas, juste les items qu'on a déjà générés
    return [
        {
            "question": item["question"],
            "answer": item["answer"],
            "contexts": item["contexts"],
            "ground_truth": item["ground_truth"],
            "retrieved_sources": item["retrieved_sources"],
            "expected_source": item["source"],
        }
        for item in dataset["data"]
        if is_generated(item)
    ]


def flush(dataset: dict) -> None:
    save_json(RESULTS_PATH, build_results(dataset))


def main() -> None:
    service = build_chat_service()

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    items = dataset["data"] if LIMIT is None else dataset["data"][:LIMIT]
    total = len(items)
    processed = 0

    for i, item in enumerate(items, start=1):
        # déjà fait -> on passe au suivant (pratique si on relance)
        if is_generated(item):
            print(f"[{i}/{total}] déjà rempli, on saute")
            continue

        print(f"[{i}/{total}] {item['question']}")
        try:
            reponse = service.answer_prompt(item["question"])
        except Exception as exc:  # rate limit, timeout, ce genre de trucs
            print(f"  ! échec: {exc}. Progression sauvegardée, arrêt.")
            flush(dataset)
            raise

        item["answer"] = reponse["answer"]
        item["contexts"] = reponse["contexts"]
        item["retrieved_sources"] = reponse["sources"]

        processed += 1
        if processed % SAVE_EVERY == 0:
            flush(dataset)

        time.sleep(1)

    flush(dataset)
    print(f"Done. {processed} nouveaux items générés, écrits dans {RESULTS_PATH} "
          f"(source {DATASET_PATH} inchangée).")


if __name__ == "__main__":
    main()
