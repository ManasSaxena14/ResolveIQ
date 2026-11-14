# Data

## Knowledge base (`data/kb/`)

The knowledge base is the help center of **Altura**, a fictional online electronics
retailer. It contains 31 articles across orders, shipping, returns, billing, accounts,
membership, warranty, support, policies and products, stored as Markdown, HTML and plain
text to exercise every supported loader.

The articles are **synthetic**. They were written for this project so that every answer
in the RAG evaluation set has a known ground truth. They do not describe any real company,
and results measured on them do not prove performance on a real company's documentation.

To use a different knowledge base, replace the contents of `data/kb/` (PDF, TXT, Markdown
and HTML are supported) and rebuild the index with `make index`. Check that the license of
any public documentation allows reuse before copying it into the repository.

## Ticket dataset (Phase 2)

The labeled ticket dataset used for the classifier and the escalation risk model is
downloaded into `data/raw/`, which is excluded from version control. Download
instructions, the dataset version and its license will be documented here in Phase 2.
