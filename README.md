# PFE Legislation Service

Service RAG (Retrieval Augmented Generation) qui répond à des questions sur la législation canadienne d'import et export de marchandises, en s'appuyant uniquement sur un corpus de documents législatifs officiels (mémorandums de l'ASFC, chapitres tarifaires, règlements sur la salubrité des aliments).

## Place dans le projet de fin d'études

Ce dépôt ne constitue pas l'application complète: il s'agit d'un des trois composants d'un projet de fin d'études (PFE) portant sur l'import et l'export de marchandises.

| Composant | Technologie | Rôle |
| --- | --- | --- |
| Frontend | React et TypeScript (Vite) | Interface utilisateur de l'application. |
| Web API | .NET | Cœur du projet: logique métier, persistance (PostgreSQL) et orchestration des appels. C'est ce service qui consomme le service RAG. |
| Service RAG (ce dépôt) | FastAPI (Python) | Couche d'intelligence documentaire: ingestion du corpus législatif, recherche hybride, reranking et génération des réponses. |

Le service RAG est conçu pour être appelé par la Web API .NET plutôt que directement par le frontend. Il est autonome et peut être démarré, ingéré et interrogé isolément, ce qui permet de le développer et de le tester sans dépendre des deux autres composants.

Le service expose une API REST (FastAPI) qui couvre les deux phases d'un RAG:

- **Ingestion**: les documents Markdown sont découpés en chunks selon leur structure de sections, vectorisés en dense (OpenAI `text-embedding-3-large`) et en sparse (BM25), puis stockés dans Qdrant.
- **Retrieval et génération**: la requête de l'utilisateur déclenche une recherche hybride (dense et sparse) dans Qdrant, les candidats sont reclassés par un reranker (Cohere `rerank-v4.0-fast`), et les meilleurs extraits sont transmis à Claude Sonnet 4.6 qui produit la réponse finale accompagnée de ses sources.

> Le fonctionnement détaillé du pipeline, l'algorithme de découpage, les stratégies de retrieval et l'analyse complète des coûts sont documentés dans [Documentation.md](Documentation.md). Le présent README se limite à la mise en route du projet.

## Pile technologique

| Couche | Technologie |
| --- | --- |
| API | FastAPI (Python 3.14), documentation interactive via Scalar |
| Base vectorielle | Qdrant (vecteurs dense et sparse nommés sur un même point) |
| Embedding dense | OpenAI `text-embedding-3-large` (3072 dimensions) |
| Embedding sparse | BM25 via `fastembed`, exécuté localement |
| Reranking | Cohere `rerank-v4.0-fast` |
| Génération | Anthropic Claude Sonnet 4.6 (sorties structurées via Pydantic) |
| Gestion des dépendances | `uv` |

## Prérequis

- [uv](https://docs.astral.sh/uv/), qui installe et gère Python 3.14 ainsi que les dépendances
- [Docker](https://www.docker.com/) et Docker Compose, si vous souhaitez faire tourner Qdrant en local
- Des clés d'API valides pour Anthropic, OpenAI et Cohere

## Démarrage

### 1. Cloner le dépôt

```bash
git clone https://github.com/PFE011-svo-import-export/PFELegislationService.git
cd PFELegislationService
```

### 2. Configurer les variables d'environnement

Copiez le fichier d'exemple, puis renseignez vos clés:

```bash
cp .env.example .env
```

Le fichier [.env.example](.env.example) liste toutes les variables attendues:

| Variable | Requis | Description |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | Oui | Clé d'API Anthropic, utilisée pour la génération de la réponse. |
| `ANTHROPIC_BASE_URL` | Non | URL de l'API Anthropic. Par défaut `https://api.anthropic.com`. |
| `OPENAI_API_KEY` | Oui | Clé d'API OpenAI, utilisée pour la vectorisation dense. |
| `OPENAI_API_MODEL` | Non | Modèle d'embedding. Par défaut `text-embedding-3-large`. |
| `COHERE_API_KEY` | Oui | Clé d'API Cohere, utilisée pour le reranking. |
| `QDRANT_HOST` | Oui | `localhost` en développement, ou le host du cluster sur Qdrant Cloud. |
| `QDRANT_PORT` | Non | Port de Qdrant. Par défaut `6333`. |
| `QDRANT_API_KEY` | Selon le cas | Obligatoire sur Qdrant Cloud, inutile pour une instance locale. |
| `QDRANT_USE_HTTPS` | Non | `true` sur Qdrant Cloud, `false` en local. Par défaut `false`. |
| `DOCUMENTS_PATH` | Non | Dossier des documents à ingérer. Par défaut `app/documents`. |

Le fichier `.env` contient des secrets et ne doit jamais être versionné.

### 3. Démarrer Qdrant

Si vous utilisez un cluster Qdrant Cloud, passez directement à l'étape suivante. Sinon, lancez une instance locale:

```bash
docker compose up -d legislation-vector-db
```

Qdrant écoute alors sur `http://localhost:6333`. La collection `documents-legislatives-import-export` est créée automatiquement au démarrage de l'API si elle n'existe pas encore.

### 4. Lancer l'API

```bash
uv sync
uv run fastapi dev
```

L'API est disponible sur `http://localhost:8000`:

- `http://localhost:8000/scalar`: documentation interactive de l'API (Scalar)
- `http://localhost:8000/docs`: documentation Swagger

### 5. Ingérer les documents

La collection est vide au premier démarrage. Lancez le pipeline d'ingestion, qui traite tous les fichiers Markdown du dossier `app/documents` en ignorant ceux déjà présents dans Qdrant:

```bash
curl -X POST http://localhost:8000/api/v1/rag/ingest
```

L'opération prend environ trois minutes pour l'ensemble du corpus. Une fois terminée, le service est prêt à répondre aux questions.

> **Formats supportés.** Le pipeline d'ingestion ne prend en charge que le format Markdown pour le moment: seuls les fichiers `.md` du dossier `app/documents` sont traités, l'algorithme de découpage s'appuyant sur la hiérarchie des titres du document. Si vos sources sont dans un autre format (PDF, DOCX, HTML, pages web), vous pouvez les convertir au préalable avec [Docling](https://github.com/docling-project/docling), une librairie qui transforme la plupart des formats de documents en Markdown exploitable par un RAG. Le projet étant toujours en cours, l'intégration de Docling directement dans le pipeline d'ingestion est prévue prochainement.

### Exécution avec Docker

L'application complète (API et base vectorielle) peut également être démarrée en une seule commande:

```bash
docker compose up --build
```

## Endpoints

### Chat (`/api/v1/legislation`)

| Méthode | Route | Description |
| --- | --- | --- |
| `POST` | `/generate` | Répond à une question libre en langage naturel et retourne la réponse ainsi que ses sources. |
| `GET` | `/tarifs/{pays}` | Retourne les traitements tarifaires applicables à un pays, sous forme structurée. |
| `GET` | `/exigences/{produit}/{pays}` | Retourne les exigences d'importation pour un produit et un pays donnés (emballage de bois, justification de l'origine, étiquetage, salubrité, etc.). |

Exemple:

```bash
curl -X POST http://localhost:8000/api/v1/legislation/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Quelles sont les exigences de marquage pour importer du café au Canada?"}'
```

### RAG (`/api/v1/rag`)

| Méthode | Route | Description |
| --- | --- | --- |
| `POST` | `/ingest` | Exécute le pipeline d'ingestion sur le dossier de documents. |
| `GET` | `/check/{filename}` | Indique si un document a déjà été ingéré dans la collection. |
| `POST` | `/compare` | Compare les résultats des trois stratégies de retrieval (dense, hybride, hybride avec reranking) pour un prompt donné et écrit un rapport Markdown. Utile pour évaluer la qualité du retrieval. |
| `DELETE` | `/collections` | Supprime la collection Qdrant. Opération destructive, une réingestion complète est ensuite nécessaire. |

## Structure du projet

```
PFELegislationService/
├── main.py                          # Point d'entrée FastAPI: middlewares, routeurs, gestion d'erreurs
├── pyproject.toml                   # Dépendances du projet (gérées par uv)
├── Dockerfile                       # Image de production de l'API
├── docker-compose.yml               # API et instance Qdrant locale
├── .env.example                     # Modèle de configuration à copier vers .env
├── Documentation.md                 # Documentation technique du RAG et analyse des coûts
│
├── app/
│   ├── api/
│   │   ├── dependencies.py          # Injection des services (RagService, ChatService)
│   │   └── routes/
│   │       ├── chat_routes.py       # Endpoints de questions et réponses
│   │       └── rag_routes.py        # Endpoints d'ingestion et d'administration de la collection
│   │
│   ├── core/
│   │   ├── config.py                # Configuration typée, chargée depuis .env (pydantic-settings)
│   │   └── dependencies.py          # Clients externes mis en cache (Anthropic, OpenAI, Cohere, Qdrant)
│   │
│   ├── services/
│   │   ├── rag_service.py           # Découpage, embeddings, recherche hybride et reranking
│   │   └── chat_service.py          # Construction du prompt augmenté et appel au LLM
│   │
│   ├── storage/
│   │   └── qdrant_vectordb.py       # Accès à Qdrant: collection, stockage, recherche dense et hybride
│   │
│   ├── models/
│   │   ├── chunk_schema.py          # Schéma d'un chunk et de ses métadonnées
│   │   ├── TraitementTarifiaire.py  # Schéma de sortie structurée
│   │   └── ExigencesImportation.py  # Schéma de sortie structurée
│   │
│   └── documents/                   # Corpus législatif en Markdown, source de l'ingestion
│
└── docs/                            # Diagrammes et images de la documentation
```

## Licence

Voir [LICENSE](LICENSE).
