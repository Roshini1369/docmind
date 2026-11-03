# AI Agents Primer

## Retrieval Augmented Generation (RAG)
Retrieval Augmented Generation combines a language model with an external knowledge base. Documents are split into chunks, converted into vector embeddings and stored in a vector database. At question time the system embeds the query, finds the most similar chunks and passes them to the model as context. This reduces hallucination and lets the model answer from private or recent data without fine-tuning.

## Context Engineering
Context engineering is the practice of deciding what information goes into a model's context window. Good context engineering removes duplicate passages, ranks content by relevance, respects a token budget and summarises old conversation turns. A model with a small, clean context usually outperforms one given a large, noisy context.

## Multi-Agent Systems
A multi-agent system splits a task between specialised agents, for example a planner, a researcher, a writer and a critic. Each agent has a narrow prompt and clear responsibilities. Frameworks such as LangGraph model the workflow as a graph of nodes and edges, which makes loops such as write, critique and revise easy to express and debug.

## Model Context Protocol (MCP)
The Model Context Protocol is an open standard that lets AI applications connect to tools and data sources through a common interface. An MCP server exposes tools, resources and prompts. Any MCP client, such as Claude Desktop, can discover and call them. This avoids writing a custom integration for every application.

## Fine-Tuning versus RAG
Fine-tuning changes model weights to teach style or behaviour and is expensive to update. RAG keeps knowledge outside the model, so updating a document immediately updates the answers. Most teams start with RAG and fine-tune only when prompting and retrieval are not enough.

## Quantization and Distillation
Quantization stores model weights in fewer bits, such as 8-bit or 4-bit, to cut memory use and speed up inference with a small accuracy cost. Distillation trains a small student model to imitate a larger teacher model, producing a cheaper model that keeps much of the quality.
