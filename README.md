<!-- ![KAPipe logo](./images/kapipe_logo_v01.png) -->

# KAPipe

**KAPipe** is a modular framework for building ***Knowledge Acquisition Systems*** from unstructured data.

KAPipe decomposes knowledge acquisition into four main stages:

1. **Extraction**: extracting knowledge units from unstructured data.
2. **Organization**: organizing extracted knowledge units into structured representations such as knowledge graph.
3. **Retrieval**: retrieving relevant knowledge for a given query or task.
4. **Utilization**: using retrieved structured knowledge for downstream tasks such as question answering.

![An overview of knowledge acquisition system](./images/knowledge_acquisition_systems_overview_figure002.png)

KAPipe is used in the following papers:

- [Nishida et al., TACL 2026, **Dissecting GraphRAG: A Modular Analysis of Knowledge Structuring for Factoid Question Answering**.](https://aclanthology.org/2026.tacl-1.29/)
- [Oumaima and Nishida et al., BioNLP 2024, **Mention-Agnostic Information Extraction for Ontological Annotation of Biomedical Articles**.](https://aclanthology.org/2024.bionlp-1.37/)

![An example of graph-based RAG architecture](./images/nishida_et_al_tacl_2026.png)

## Installation

```bash
python -m pip install -U kapipe
```

For local development:

```bash
git clone https://github.com/norikinishida/kapipe.git
cd kapipe
python -m pip install -e .
```

Some pretrained models and configuration files are distributed separately.

```bash
mkdir -p ~/.kapipe
mv release.YYYYMMDD.tar.gz ~/.kapipe
cd ~/.kapipe
tar -zxvf release.YYYYMMDD.tar.gz
```

Release files are available here:
Please use the latest release file!

[KAPipe Release Files](https://drive.google.com/drive/folders/16ypMCoLYf5kDxglDD_NYoCNAfhTy4Qwp)

## Components

In KAPipe, a ***component*** is a modular processing unit that implements a specific approach within one of the four stages: extraction, organization, retrieval, or utilization.

The following table summarizes the components currently supported by KAPipe.

| Stage | Component | Module | Docs | Example |
|---|---|---|---|---|
| Extraction | Named Entity Recognition | `kapipe.ner` | [Docs](docs/components/ner.md) | [Example](experiments/ner) |
| Extraction | Entity Disambiguation (Retrieval) | `kapipe.ed_retrieval` | [Docs](docs/components/ed_retrieval.md) | [Example](experiments/ed_retrieval) |
| Extraction | Entity Disambiguation (Reranking) | `kapipe.ed_reranking` | [Docs](docs/components/ed_reranking.md) | [Example](experiments/ed_reranking) |
| Extraction | Document-level Relation Extraction | `kapipe.docre` | [Docs](docs/components/docre.md) | [Example](experiments/docre) |
| Organization | Entity Graph Construction | `kapipe.entity_graph_construction` | [Docs](docs/components/entity_graph_construction.md) | [Example](experiments/entity_graph_construction) |
| Organization | Community Clustering | `kapipe.community_clustering` | [Docs](docs/components/community_clustering.md) | [Example](experiments/community_clustering) |
| Organization | Report Generation | `kapipe.report_generation` | [Docs](docs/components/report_generation.md) | [Example](experiments/report_generation) |
| Organization | Chunking | `kapipe.chunking` | [Docs](docs/components/chunking.md) | [Example](experiments/chunking) |
| Retrieval | Passage Retrieval | `kapipe.passage_retrieval` | [Docs](docs/components/passage_retrieval.md) | [Example](experiments/passage_retrieval) |
| Utilization | Question Answering | `kapipe.qa` | [Docs](docs/components/qa.md) | [Example](experiments/qa) |

## Pipelines

Pipelines (`kapipe.pipelines`) are convenience classes for chaining components that are commonly used together.
Internally, a pipeline connects the outputs of one component to the inputs of the next component.

| Pipeline | Description | Docs | Example |
|---|---|---|---|
| `TripleExtractionPipeline` | Chains NER, Entity Disambiguation (Retrieval), Entity Disambiguation (Reranking), and Document-level Relation Extraction components | [Docs](docs/pipelines/triple_extraction_pipeline.md) | [Example](experiments/triple_extraction_pipeline) |
| `RAGPipeline` | Chains Passage Retrieval and Question Answering components | [Docs](docs/pipelines/rag_pipeline.md) | [Example](experiments/rag_pipeline) |
| `GraphRAGPipeline` | Chains triple extraction, Entity Graph Construction, Community Clustering, Report Generation, Passage Retrieval, and Question Answering components | [Docs](docs/pipelines/graphrag_pipeline.md) | [Example](experiments/graphrag_pipeline_tacl2026) |

## Agents

Agents (`kapipe.agents`) use an LLM to dynamically reason, select Tools, observe Tool-call results, and generate a final response.
Unlike pipelines, which connect components in a predefined sequence, agents decide which action to take based on the request and the execution trajectory.
A Tool can wrap a KAPipe component or any other callable function.

| Agent | Description | Docs | Example |
|---|---|---|---|
| `ToolCallingAgent` | Performs ReAct-style inference by repeatedly calling Tools and observing their results until it generates a final answer | [Docs](docs/agents/tool_calling_agent.md) | [Example](experiments/agentic_search) |

## Quickstart

### Example 1

This example shows how to instantiate Passage Retrieval and QA components and run Retrieval-Augmented Generation (RAG).

```python
import os

from kapipe import utils
from kapipe.llms import OpenAILLM
from kapipe.passage_retrieval import Qwen3Embedding
from kapipe.qa import LLMQA


# Set input and output paths
data_dir = "experiments/passage_retrieval/data/examples"
index_dir = "./indexes"

# Load passages and questions
passages = utils.read_jsonl(os.path.join(data_dir, "passages.jsonl"))
questions = utils.read_json(os.path.join(data_dir, "questions.json"))

# Instantiate the Passage Retrieval component
passage_retrieval = Qwen3Embedding(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    max_passage_length=8192,
    normalize=True,
    metric="inner-product",
    query_instruction="Given a question, retrieve relevant passages that answer the question.",
)

# Instantiate the QA component
llm = OpenAILLM(model_name="gpt-5.4-nano", max_new_tokens=8192)
qa = LLMQA(
    model=llm,
    prompt_template_name_or_path="qa_03_with_context",
)

# Build a retrieval index over passages
passage_retrieval.make_index(
    passages=passages,
    index_dir=index_dir,
    batch_size=64,
)

# Answer questions by chaining retrieval and QA
answers = []
for question in questions:

    # Retrieve relevant passages
    retrieved_passages = passage_retrieval.search(
        queries=[question["question"]],
        top_k=5,
    )[0]

    # Wrap retrieved passages in the QA input format
    contexts_for_question = {
        "question_key": question["question_key"],
        "contexts": retrieved_passages,
    }

    # Generate an answer
    answer = qa.answer(
        question=question,
        contexts_for_question=contexts_for_question,
    )

    # Preserve retrieved contexts
    answer["contexts"] = retrieved_passages

    answers.append(answer)

# Save the results
utils.write_json("./predictions.json", answers)
```

### Example 2

This example shows how to instantiate `GraphRAGPipeline`, structure knowledge, and run inference with GraphRAG.

The full executable version is available in [`experiments/graphrag_pipeline_tacl2026`](experiments/graphrag_pipeline_tacl2026).

```python
import os

from kapipe import utils
from kapipe.pipelines import GraphRAGPipeline
from kapipe.llms import OpenAILLM
from kapipe.ner import LLMNER
from kapipe.ed_retrieval import BlinkBiEncoder
from kapipe.ed_reranking import LLMED
from kapipe.docre import LLMDocRE
from kapipe.entity_graph_construction import EntityGraphConstructor
from kapipe.community_clustering import NeighborhoodAggregation
from kapipe.report_generation import TemplateBasedReportGenerator
from kapipe.chunking import Chunker
from kapipe.passage_retrieval import Qwen3Embedding
from kapipe.qa import LLMQA


# Set input and output paths
data_dir = "experiments/graphrag_pipeline_tacl2026/data/examples"
index_dir = "./indexes"

# Instantiate the components
llm = OpenAILLM(model_name="gpt-5.4-nano", max_new_tokens=8192)
ner = LLMNER.from_identifier(llm, "llm_ner_cdr")
ed_retrieval = BlinkBiEncoder.from_identifier("blink_bi_encoder_cdr")
ed_retrieval.make_index(use_precomputed_entity_vectors=True)
ed_reranking = LLMED.from_identifier(llm, "llm_ed_cdr")
docre = LLMDocRE.from_identifier(llm, "llm_docre_cdr")
entity_graph_construction = EntityGraphConstructor()
community_clustering = NeighborhoodAggregation(hop_size=1)
report_generation = TemplateBasedReportGenerator()
chunker = Chunker(model_name="en_core_sci_md")
passage_retrieval = Qwen3Embedding(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    max_passage_length=8192,
    normalize=True,
    metric="inner-product",
    query_instruction="Given a question, retrieve relevant passages that answer the question."
)
qa = LLMQA(
    model=llm,
    prompt_template_name_or_path="qa_03_with_context",
)

# Instantiate the GraphRAG pipeline
graphrag = GraphRAGPipeline(
    ner=ner,
    ed_retrieval=ed_retrieval,
    ed_reranking=ed_reranking,
    docre=docre,
    entity_graph_construction=entity_graph_construction,
    community_clustering=community_clustering,
    report_generation=report_generation,
    chunker=chunker,
    passage_retrieval=passage_retrieval,
    qa=qa,
)

# Step 1. Extract triples from documents
documents = utils.read_json(os.path.join(data_dir, "documents.json"))
graphrag.extract_triples(
    documents=documents,
    retrieval_size=10,
    index_dir=index_dir,
)

# Step 2. Construct an entity graph from triples
graph = graphrag.construct_entity_graph(
    documents_path_list=[os.path.join(index_dir, "documents_with_triples.json")],
    entity_dict_path=os.path.join(data_dir, "entity_dict.json"),
    additional_triples_path=None,
    index_dir=index_dir,
)

# Step 3. Cluster the graph into communities (subgraphs)
communities = graphrag.cluster_communities(
    graph=graph,
    index_dir=index_dir,
)

# Step 4. Generate reports for each community
reports = graphrag.generate_community_reports(
    graph=graph,
    communities=communities,
    index_dir=index_dir,
)

# Step 5. Chunk community reports into chunks
chunked_reports = graphrag.chunk_reports(
    reports=reports,
    window_size=100,
    index_dir=index_dir,
)

# Step 6. Build a retrieval index over the chunked reports
graphrag.make_passage_retrieval_index(
    chunked_reports=chunked_reports,
    batch_size=64,
    index_dir=index_dir,
)

# Step 7. Load the retrieval index and answer questions
graphrag.load_passage_retrieval_index(index_dir=index_dir)
questions = utils.read_json(os.path.join(data_dir, "questions.json"))
answers = [
    graphrag.infer(question=question, top_k=5)
    for question in questions
]

# Save the results
utils.write_json("./predictions.json", answers)
```

The components can also be used independently.
Please see the corresponding documentation for each component for more details.

## Citation / Publication

If **KAPipe** is helpful for your work, please consider citing the following paper:

**Dissecting GraphRAG: A Modular Analysis of Knowledge Structuring for Factoid Question Answering**.
Noriki Nishida, Rumana Ferdous Munne, Shanshan Liu, Narumi Tokunaga, Yuki Yamagata, Fei Cheng, Kouji Kozaki, and Yuji Matsumoto.
Transactions of the Association for Computational Linguistics (TACL), vol. 14, pp. 627-655. 2026.
(Presented at ACL 2026)

```bibtex
@article{nishida-etal-2026-dissecting,
    title = "Dissecting {G}raph{RAG}: A Modular Analysis of Knowledge Structuring for Factoid Question Answering",
    author = "Nishida, Noriki  and
      Munne, Rumana Ferdous  and
      Liu, Shanshan  and
      Tokunaga, Narumi  and
      Yamagata, Yuki  and
      Cheng, Fei  and
      Kozaki, Kouji  and
      Matsumoto, Yuji",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "14",
    year = "2026",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2026.tacl-1.29/",
    doi = "10.1162/tacl.a.615",
    pages = "627--655"
}
```

