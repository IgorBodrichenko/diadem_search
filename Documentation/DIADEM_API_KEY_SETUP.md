# Diadem API Key Setup Documentation

## Required Keys

This project needs several environment variables to run.

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
PINECONE_HOST=
```

Optional Bubble integration:

```env
BUBBLE_API_BASE=
BUBBLE_API_KEY=
```

PDF ingestion also uses:

```env
PINECONE_ENV=
```

## What Each Key Does

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Used to create embeddings for search. |
| `ANTHROPIC_API_KEY` | Used to generate Claude chat/coaching responses. |
| `PINECONE_API_KEY` | Used to connect to Pinecone vector storage. |
| `PINECONE_INDEX_NAME` | The name of the Pinecone index containing Diadem content. |
| `PINECONE_HOST` | The host URL for the specific Pinecone index. |
| `BUBBLE_API_BASE` | Optional Bubble Data API base URL. |
| `BUBBLE_API_KEY` | Optional Bubble API token for reading template data. |
| `PINECONE_ENV` | Used by the PDF ingestion script. |

## Where To Get The Keys

### OpenAI

Go to [OpenAI Platform](https://platform.openai.com/).

Create or open a project, then create an API key.

Use it as:

```env
OPENAI_API_KEY=
```

### Anthropic / Claude

Go to [Claude Console](https://platform.claude.com/).

Create an API key from the Anthropic Console.

Use it as:

```env
ANTHROPIC_API_KEY=
```

### Pinecone

Go to [Pinecone Console](https://app.pinecone.io/).

You need:

```env
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
PINECONE_HOST=
```

The API key comes from your Pinecone project settings.

The index name and host come from the specific Pinecone index page.

Official docs: [Pinecone API keys](https://docs.pinecone.io/guides/projects/manage-api-keys)

### Bubble

Only needed if the backend should read Bubble template state.

Go to your Bubble app settings and enable the Data API.

Use:

```env
BUBBLE_API_BASE=
BUBBLE_API_KEY=
```

Official docs: [Bubble Data API](https://manual.bubble.io/core-resources/api/the-bubble-api/the-data-api)

## Where To Put The Keys Locally

Create a `.env` file inside the project folder:

```text
C:\Users\lukeb\Desktop\diadem_search-test_v3\diadem_search-test_v3\.env
```

Example:

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
PINECONE_HOST=

BUBBLE_API_BASE=
BUBBLE_API_KEY=
```

For ingestion:

```env
PINECONE_ENV=
```

## Where To Put The Keys On Render

In Render:

```text
Service -> Environment -> Environment Variables
```

Add:

```env
OPENAI_API_KEY
ANTHROPIC_API_KEY
PINECONE_API_KEY
PINECONE_INDEX_NAME
PINECONE_HOST
```

Optional:

```env
BUBBLE_API_BASE
BUBBLE_API_KEY
PINECONE_ENV
```

## Important Notes

Do not paste keys directly into `app.py`.

Do not commit `.env` to GitHub.

The current app will not run with only an OpenAI key. It needs OpenAI, Anthropic, and Pinecone configured.
