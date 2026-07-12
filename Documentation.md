# Documentation du RAG (Retrieval augmented generation)

## Qu'est ce qu'un RAG?

Retrieval-Augmented Generation (RAG) est une architecture (ou technique) qui améliore la capacité d'un LLM (Large language model) en le connectant à une source de connaissance externe (base de données, documents, APIs, etc). Au lieu de se baser uniquement sur l'information que le LLM a été entrainé dessus, le modèle récupère les éléments pertinents et mis à jour de ces sources et l'intègrent dans le processus de génération. Cela permet à un LLM de produire une réponse plus précise, mis à jour, et spécifique à un domaine sans la nécéssité de l'entrainer.

## Problèmes que le RAG vient résoudre dans le cadre de notre projet?

## Les 2 phases du RAG

RAG est divisé en deux phases importantes et complémentaire. La phase d'ingestion des données est le processus dont les sources d'informations sont transformés et stocké dans la base de données vectorielle. La phase de retrieval est le processus dont on récupère les meilleurs extraits d'informations et pertinent pour générer une réponse.

### Phase Ingestion (aussi appelé 'indexing pipeline')

La phase d'ingestion transforme les sources de données brutes en vecteurs interrogeables, stockés dans la base de données vectorielle. Elle se déroule en plusieurs étapes séquentielles:

1. **Sources** collecte des données brutes depuis les différentes sources d'information. Celles-ci peuvent être sous plusieurs formes comme: PDF, HTML, texte brute, etc. Dans le cadre de ce projet, les sources sont des documents legislatives sur le domaine d'import et export de la marchandise en format PDF.
2. **Transformation des documents** cette étape transforme les documents dans un format plus compatible avec RAG (dans le cas de ce projet, il s'agit du format Markdown). La transformation des documents en format markdown est faite à l'aide d'un LLM. Parcontre, un autre très bonne option gratuit est la librairie Docling qui est capable de transformer n'importe quel type de source d'information (meme un site web) en markdown.
3. **Découpage (chunking)** les documents sont découpés en segments de taille raisonnable (chunks), afin de respecter la limite de contexte des modèles d'embedding et de garder chaque extrait suffisamment précis et autonome pour être pertinent lors du retrieval. L'algorithme utilisé pour le processus de découpage basé sur la strcutrue du document ([voir l'algorithme de découpage](#algorithme-de-découpage---découpage-basé-sur-la-structure-du-document))
4. **Embeddings** chaque chunk est transformé en **deux représentations vectorielles complémentaires**, générées en parallèle:
    - **Vecteur dense**: le modèle d'embedding utilisé est le 'text-embedding-3-large' de OpenAI qui crée des vecteurs de dimension de 3072. Un vecteur dense capture le sens sémantique du chunk dans un espace continu de quelques centaines à quelques miliers de dimensions.
    - **Vecteur sparse**: produit par une méthode comme BM25 ou SPLADE qui capture la présence et l'importance des mots-clés spécifiques du chunk dans un espace de très haute dimensionnalité, dizaine de miliers (majoritairement des zéros).
5. **Stockage** les deux vecteurs (dense et sparse) sont stockés comme deux vecteurs nommés sur le **même point** dans Qdrant. Cela évite de dupliquer les chunks et permet, plus tard, d'effectuer une recherche hybride (voir section [Recherche hybride](#hybrid-search)).

Cette double vectorisation à l'ingestion est ce qui rend possible, côté retrieval, la combinaison entre recherche sémantique (dense) et recherche par mots-clés (sparse) — chacune compensant les angles morts de l'autre (voir _Dense vector_, _Sparse search_, _Hybrid search_ plus bas).

#### Algorithme de découpage - découpage basé sur la structure du document

Découpage basé sur la structure découpe un document selon des sections au lieu d'un nombre de charactères fixe. Si une section dépasse la taille limit d'un chunk, l'algorithme va diviser la section en plusieurs sections en gardant le titre pour chacun. Cet algorithme est utilisé parce que les documents legislativs suivent une hiérarchie par sections qui intègre des structures strictes tels que des tableaux.

![Description de l'image](docs/Images/RAG%20-%20Ingestion%20phase.jpg)

### Phases Retrieval - trois étapes

#### Retrieval - techniques/stratégies

Ce sont des techniques ou des stratégies qui améliorent la qualité du 'retrieval' des extraits d'informations de la base de données vectorielle. Le 'Retrieval' est une étape très importante pour générer la bonne réponse à l'utilisateur parce que les extraits de textes sont ensuite jumelés avec le prompt initiale de l'utilisateur pour créer un prompt final à envoyer à un LLM. Si ces extraits de texte ne sont pas précis, la réponse du LLM ne le sera pas non plus. Les models de embeddings (transformation du text en représentaiton numérique) se situent dans trois catégories: Dense vector (Embeddings), Sparse vector (BM25 ou Keyword), et Hybrid (combinaison des deux)

##### Dense vector

Utilisé pour la compréhension et recherche sémantique.

#### Sparse search

Utilisé pour la compréhension et recherche de mots clés

##### Hybrid search

##### Reranking

## Pile technologique utilisé

## Diagrames

### Diagramme déploiement

## Les coûts associés

### Évaluation des coûts services externe vs open source

#### Coûts d'utilisation des services externe

#### Coûts de déploiment et infra pour open source (Ollama)

#### Verdict

### Phase ingestion complet

### Prompt + réponse
