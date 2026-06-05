# Systèmes de Recommandation : Architecture, Données et Évaluation

> **Document de test** — Conçu pour valider le découpage sémantique (chunking) et la génération d'embeddings dans un pipeline RAG. Contient des sections hiérarchiques, des tableaux structurés, des listes, des données numériques et des blocs de code.

---

## 1. Introduction aux Systèmes de Recommandation

Un **système de recommandation** est un sous-classe de système de filtrage de l'information qui cherche à prédire la *préférence* qu'un utilisateur attribuerait à un élément. Ces systèmes sont omniprésents dans les plateformes modernes : commerce électronique, streaming vidéo, réseaux sociaux, et moteurs de recherche.

Les systèmes de recommandation reposent sur trois paradigmes fondamentaux :

1. **Filtrage collaboratif (Collaborative Filtering)** — exploite les comportements d'utilisateurs similaires pour inférer des préférences.
2. **Filtrage basé sur le contenu (Content-Based Filtering)** — compare les attributs des items aux préférences passées de l'utilisateur.
3. **Approches hybrides** — combinent plusieurs méthodes pour pallier les limitations individuelles (cold start, sparsité).

> **Concept clé — Cold Start** : Problème survenant quand un nouvel utilisateur ou un nouvel item n'a aucun historique d'interaction. Le filtrage collaboratif pur est particulièrement vulnérable à ce phénomène.

---

## 2. Comparaison des Algorithmes

### 2.1 Tableau comparatif des méthodes classiques

| Algorithme | Type | Complexité entraînement | Scalabilité | Cold Start | RMSE typique |
|---|---|---|---|---|---|
| User-Based KNN | Collaboratif | O(U² · I) | Faible | Problématique | 0.92 – 1.05 |
| Item-Based KNN | Collaboratif | O(I² · U) | Moyenne | Partiel | 0.88 – 1.02 |
| SVD (Funk) | Matriciel | O(U · I · K) | Élevée | Partiel | 0.84 – 0.96 |
| NMF | Matriciel | O(U · I · K) | Élevée | Partiel | 0.86 – 0.98 |
| ALS | Matriciel | O(U · K² + I · K²) | Très élevée | Partiel | 0.82 – 0.94 |
| Neural CF | Deep Learning | O(E · B · L) | Très élevée | Amélioré | 0.78 – 0.91 |
| Two-Tower | Deep Learning | O(E · B · L) | Très élevée | Fort | 0.76 – 0.89 |

> **Légende** : U = utilisateurs, I = items, K = facteurs latents, E = époques, B = batch size, L = couches réseau.

### 2.2 Métriques d'évaluation

Les systèmes de recommandation sont évalués selon deux axes principaux : la **précision de prédiction** et la **qualité du classement**.

| Métrique | Formule simplifiée | Interprétation | Plage |
|---|---|---|---|
| RMSE | √(Σ(r̂ᵢ − rᵢ)² / n) | Erreur quadratique moyenne | [0, ∞) → plus bas = mieux |
| MAE | Σ\|r̂ᵢ − rᵢ\| / n | Erreur absolue moyenne | [0, ∞) → plus bas = mieux |
| Precision@K | TP@K / K | Pertinence dans les K premiers | [0, 1] → plus haut = mieux |
| Recall@K | TP@K / Pertinents | Couverture des items pertinents | [0, 1] → plus haut = mieux |
| NDCG@K | DCG@K / IDCG@K | Classement pondéré par position | [0, 1] → plus haut = mieux |
| MAP | Moyenne des AP | Précision moyenne sur tous les users | [0, 1] → plus haut = mieux |
| Coverage | \|Items recommandés\| / \|Items total\| | Diversité du catalogue couvert | [0, 1] |

---

## 3. Architecture d'un Pipeline RAG Moderne

### 3.1 Composants principaux

Un pipeline **RAG (Retrieval-Augmented Generation)** moderne se décompose en plusieurs étapes distinctes :

```
Documents bruts
     │
     ▼
┌─────────────────┐
│   Prétraitement  │  → Nettoyage, normalisation, détection de langue
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chunking     │  → Découpage sémantique ou par tokens
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedding     │  → Encodage vectoriel (ex: text-embedding-3-small)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │  → Stockage + indexation (FAISS, Qdrant, Pinecone)
└────────┬────────┘
         │
    [Requête utilisateur]
         │
         ▼
┌─────────────────┐
│   Retrieval     │  → Recherche par similarité cosinus (Top-K)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Génération    │  → LLM conditionné sur les chunks récupérés
└─────────────────┘
```

### 3.2 Stratégies de chunking

Le découpage des documents est critique pour la qualité du RAG. Un mauvais chunking dégrade la précision de la récupération même avec un excellent modèle d'embedding.

| Stratégie | Description | Avantages | Inconvénients | Usage recommandé |
|---|---|---|---|---|
| Fixed-size | Découpage à N tokens fixes | Simple, rapide | Coupe le contexte sémantique | Prototypes rapides |
| Sentence-based | Découpage par phrases | Préserve la cohérence | Chunks inégaux | Textes narratifs |
| Recursive | Hiérarchie de séparateurs | Équilibré | Paramétrage délicat | Usage général |
| Semantic | Clustering par similarité | Respecte le sens | Coûteux en calcul | Documents complexes |
| Agentic | LLM détermine les coupures | Précision maximale | Très lent et cher | Données critiques |

#### Paramètres typiques de chunking

```python
# Exemple de configuration — RecursiveCharacterTextSplitter (LangChain)
chunk_size       = 512    # tokens par chunk
chunk_overlap    = 64     # tokens de chevauchement
separators       = ["\n\n", "\n", ". ", " ", ""]
length_function  = tiktoken_len  # comptage réel en tokens
```

---

## 4. Données de Benchmark — Jeux de Données Publics

### 4.1 Datasets standard pour la recommandation

| Dataset | Domaine | Utilisateurs | Items | Interactions | Densité | Source |
|---|---|---|---|---|---|---|
| MovieLens 1M | Films | 6 040 | 3 952 | 1 000 209 | 4.19% | GroupLens |
| MovieLens 25M | Films | 162 541 | 62 423 | 25 000 095 | 0.25% | GroupLens |
| Amazon Reviews | E-commerce | 43 531 | 3 618 | 1 689 188 | 0.11% | Amazon |
| Netflix Prize | Streaming | 480 189 | 17 770 | 100 480 507 | 1.17% | Netflix |
| Yelp 2023 | Restaurants | 1 987 897 | 150 346 | 6 990 280 | 0.002% | Yelp |
| LastFM | Musique | 1 892 | 17 632 | 92 834 | 0.28% | HetRec |
| BookCrossing | Livres | 278 858 | 271 379 | 1 149 780 | 0.002% | Ziegler |

### 4.2 Résultats expérimentaux — MovieLens 1M

Résultats reproductibles obtenus avec une division 80/10/10 (train/val/test), graines aléatoires fixées à 42.

| Modèle | RMSE ↓ | MAE ↓ | Precision@10 ↑ | Recall@10 ↑ | NDCG@10 ↑ | Temps entraînement |
|---|---|---|---|---|---|---|
| User-KNN (k=50) | 0.9812 | 0.7743 | 0.2134 | 0.1456 | 0.2891 | 2 min |
| Item-KNN (k=50) | 0.9234 | 0.7281 | 0.2287 | 0.1578 | 0.3012 | 4 min |
| SVD (K=100) | 0.8731 | 0.6892 | 0.2541 | 0.1723 | 0.3287 | 8 min |
| ALS (K=128) | 0.8614 | 0.6801 | 0.2678 | 0.1834 | 0.3401 | 12 min |
| Neural CF | 0.8392 | 0.6614 | 0.2812 | 0.1967 | 0.3589 | 45 min |
| Two-Tower + BM25 | **0.8201** | **0.6489** | **0.2994** | **0.2103** | **0.3712** | 1h 20 min |

---

## 5. Embeddings et Modèles de Représentation

### 5.1 Modèles d'embedding populaires

Les embeddings transforment du texte brut en vecteurs denses dans un espace latent où la proximité cosinus reflète la similarité sémantique.

| Modèle | Dimension | Tokens max | Langues | Score MTEB | Coût (1M tokens) |
|---|---|---|---|---|---|
| text-embedding-3-small | 1 536 | 8 191 | 100+ | 62.3 | $0.020 |
| text-embedding-3-large | 3 072 | 8 191 | 100+ | 64.6 | $0.130 |
| text-embedding-ada-002 | 1 536 | 8 191 | 100+ | 61.0 | $0.100 |
| all-MiniLM-L6-v2 | 384 | 256 | EN | 56.3 | Gratuit (local) |
| all-mpnet-base-v2 | 768 | 512 | EN | 57.8 | Gratuit (local) |
| multilingual-e5-large | 1 024 | 514 | 100+ | 61.5 | Gratuit (local) |
| BGE-M3 | 1 024 | 8 192 | 100+ | 63.1 | Gratuit (local) |

### 5.2 Impact de la dimension sur les performances

La **malédiction de la dimensionnalité** affecte la recherche vectorielle : au-delà d'un certain seuil, augmenter la dimension n'améliore plus la qualité de récupération mais augmente le coût mémoire et de calcul.

- **Dimensions ≤ 256** : Rapides, légers, mais pauvres en nuance sémantique. Conviennent pour des corpus homogènes et simples.
- **Dimensions 512–1024** : Bon compromis performance/coût pour la majorité des applications RAG.
- **Dimensions > 2048** : Utilisées pour des tâches de haute précision (recherche scientifique, juridique). Le gain marginal diminue au-delà de 3072.

---

## 6. Glossaire Technique

| Terme | Définition |
|---|---|
| **Embedding** | Représentation vectorielle dense d'un objet (texte, image, utilisateur) dans un espace de faible dimension |
| **Chunk** | Fragment d'un document découpé pour l'indexation dans un vector store |
| **Vector Store** | Base de données spécialisée dans le stockage et la recherche de vecteurs (ex: FAISS, Qdrant, Pinecone, Weaviate) |
| **Similarité cosinus** | Mesure de similarité entre deux vecteurs basée sur l'angle entre eux : cos(θ) = A·B / (\|A\|·\|B\|) |
| **Top-K retrieval** | Récupération des K vecteurs les plus proches d'une requête dans le vector store |
| **Reranking** | Étape de re-classement des résultats récupérés à l'aide d'un modèle plus précis (ex: cross-encoder) |
| **Sparse retrieval** | Recherche basée sur des termes exacts (BM25, TF-IDF) — complémentaire aux embeddings |
| **Hybrid search** | Combinaison de la recherche dense (embeddings) et sparse (BM25) via RRF ou pondération |
| **MTEB** | Massive Text Embedding Benchmark — référence standard pour comparer les modèles d'embeddings |
| **Hallucination** | Génération de contenu plausible mais factuellement incorrect par un LLM |
| **Grounding** | Ancrage de la réponse du LLM dans des sources factuelles récupérées (le principe du RAG) |

---

*Fin du document de test — 2 pages environ en rendu PDF standard (11pt, marges 2cm).*
