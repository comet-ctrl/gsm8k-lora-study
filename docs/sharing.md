# Sharing a run

The current source tree contains fresh project code, synthetic tests, and an output-free notebook.
It excludes source course documents, assignment notebooks, reports, submission archives, and private
notebook links. Original local materials are not needed to use the project.

Generated data and weights are ignored. Manifests use explicit fields rather than serializing command-line
paths, environment variables, account tokens, hostnames, or usernames. Evaluation summaries identify
adapter weights by checksum rather than the local adapter path.

Before sharing:

1. Review generated responses and clear notebook outputs, execution counts, and private metadata.
2. Check upstream licenses before redistributing weights or dataset content.
3. Review staged changes and commit identity. A GitHub no-reply address avoids adding a personal email.

Deleting files in a commit does not remove them from old commits, branches, pull requests, forks, or
cached copies. These protections cover the new source tree and metadata, not historical copies.
Removing sensitive historical material is a separate operation.
