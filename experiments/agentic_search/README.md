# agentic_search

This example demonstrates agentic search using a Tool-Calling Agent (`kapipe.agents.ToolCallingAgent`).
For a given request, the agent follows a Reasoning and Acting (ReAct)-style loop: it repeatedly reasons, calls a Tool (`kapipe.agents.Tool`), observes the Tool-call result, and either continues the loop or generates a final answer.

In this example, the Tool is a dense retriever, and the Tool-call result is a set of retrieved passages.
The retrieval Tool internally uses a KAPipe Passage Retrieval component (`kapipe.passage_retrieval`).
However, retrieval is only one example of Tool use; any Tool with an appropriate interface can be passed to the agent.
The agent’s LLM is provided through `kapipe.llms`.

The corpus of passages is indexed by the dense retriever in advance, independently of the Tool-Calling Agent.
The dense retriever loads the prebuilt index and is then wrapped as a Tool, which is passed to the Tool-Calling Agent.
That is, the indexed corpus (i.e., the external knowledge) is connected to the agent through the retrieval Tool.

## Step 1. Installation

Install KAPipe by following the `README.md` in the repository root.

If you use a pyenv environment, you can install the dependencies as follows.

```bash
# 1. Create your own Python environment
pyenv install 3.11.14
pyenv virtualenv 3.11.14 <your-favorite-env-name>

# 2. Activate the Python environment in this directory
cd experiments/agentic_search
pyenv local <your-favorite-env-name>

# 3. Install the KAPipe library
python -m pip install -U kapipe
# or
python -m pip install -e ../..

# 4. Install dependencies specific to this directory
python -m pip install -r requirements.txt
```

## Step 2. Dataset Preparation

### Example dataset

This directory already includes example data.

```bash
experiments/agentic_search/data/examples/corpus/passages.jsonl
experiments/agentic_search/data/examples/qa/questions.json
experiments/agentic_search/data/examples/qa/questions_with_answers.json
experiments/agentic_search/data/examples/qa/questions.gold_contexts.json
```

Each corpus and gold-context passage requires `passage_key` and `text`.

## Step 3. Configuration Setup

Adjust the settings in the following configuration file (HOCON format).

```bash
experiments/agentic_search/config/default.conf
```

The configuration specifies the Passage Retrieval component, the LLM, and the maximum number of agent steps.
The current example uses Qwen3-Embedding for passage retrieval and an OpenAI model for reasoning and response generation.

## Step 4. Experiment Running

First, check `STORAGE_DATA` and `STORAGE_RESULTS` in the execution script and adjust them to your environment.

If you modified the configuration file or added new configuration entries, make sure that the execution script refers to the intended configuration.

Build the Passage Retrieval index first.

```bash
bash ./run_passage_retrieval_indexing.sh
```

Then, make sure that `INDEX_DIR` in `run_tool_calling_agent.sh` refers to the constructed index and run the Tool-Calling Agent.

```bash
bash ./run_tool_calling_agent.sh
```

If you use the OpenAI API, set your API key in advance.
