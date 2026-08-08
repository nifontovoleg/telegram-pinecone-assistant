# Telegram Pinecone Assistant

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?logo=telegram&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-000000?logo=pinecone&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Personal Telegram assistant with **long-term semantic memory**. Every user message is embedded and stored in Pinecone; future replies are grounded in relevant past messages via vector search and OpenAI chat completion.

---

## What it does

| Capability | Description |
|---|---|
| **Persistent memory** | Stores only the user's original message text in Pinecone (`type: user_message`) |
| **Semantic recall** | Retrieves top‑K relevant messages before each reply |
| **Duplicate control** | Cosine similarity threshold (`0.85`) — `inserted` / `updated` / `skipped` |
| **Per-user isolation** | Metadata filter by `user_id` |
| **First-message greeting** | Greets by name once (`;nickname` prefix or Telegram profile) |

### Use cases

- Personal AI companion that remembers preferences and context across sessions
- Support bot prototype with searchable conversation history
- RAG-style assistant without a separate document pipeline — memory **is** the knowledge base
- Foundation for multi-tenant bots, CRM notes, or internal team assistants

---

## Architecture

```
User (Telegram)
      │
      ▼
telegram_bot.py          ← pyTelegramBotAPI (AsyncTeleBot), handlers, OpenAI chat
      │
      ├── embed (OpenAI)   ← standard index only
      │
      ▼
pinecone_manager.py      ← upsert / query / duplicate detection
      │
      ▼
Pinecone index (nemo)    ← namespace __default__
```

### Pinecone record schema

```json
{
  "id": "user_{telegram_id}_msg_{timestamp}",
  "metadata": {
    "first_name": "...",
    "language_code": "ru",
    "text": "original user message",
    "timestamp": "2026-08-08T14:01:08.890403",
    "type": "user_message",
    "user_id": "5921878055",
    "username": "handle"
  }
}
```

---

## Project layout

```
telegram-pinecone-assistant/
├── telegram_bot.py      # Bot entry point
├── pinecone_manager.py    # Pinecone read/write layer
├── EnvExample           # Environment template
├── requirements.txt
└── README.md
```

---

## Requirements

- Python **3.11+**
- [Telegram Bot Token](https://t.me/BotFather)
- [Pinecone](https://www.pinecone.io/) account + index
- [OpenAI API key](https://platform.openai.com/) (chat + embeddings)

### Pinecone index

| Mode | `PINECONE_INTEGRATED` | Embeddings |
|---|---|---|
| **Standard** (default) | `false` | OpenAI `text-embedding-3-small` (1536 dims) |
| **Integrated** | `true` | Pinecone server-side |

Create a **standard** index with **dimension = 1536** for the default embedding model.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/nifontovoleg/telegram-pinecone-assistant.git
cd telegram-pinecone-assistant
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp EnvExample .env
```

Fill in `.env`:

```env
TELEGRAM_BOT_TOKEN=your-bot-token
OPENAI_API_KEY=your-openai-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=nemo
PINECONE_INTEGRATED=false
EMBEDDING_MODEL=text-embedding-3-small
MEMORY_TOP_K=8
```

> Never commit `.env` — it is listed in `.gitignore`.

### 3. Run the bot

```bash
python telegram_bot.py
```

Expected log:

```text
Бот инициализирован успешно
Запуск бота...
Индекс Pinecone: nemo (standard)
Сообщение сохранено: user_... action: inserted
```

### 4. Test Pinecone layer directly

```bash
python pinecone_manager.py
```

Output format:

```python
{'matches': [{'id': '...', 'metadata': {...}, 'score': 0.83, 'values': []}, ...]}
```

---

## Bot commands

| Command | Action |
|---|---|
| `/start` | Welcome message |
| `/help` | Usage hints |
| `/memory [query]` | Show semantically relevant stored messages |
| `/forget` | Delete all vectors for current user |

---

## How to extend

### Change personality

Set `SYSTEM_PROMPT` in `.env` or edit the default in `telegram_bot.py` → `load_settings()`.

### Store more metadata

Extend `_build_metadata()` in `telegram_bot.py` — add fields, then use them in Pinecone filters or LLM context.

### Switch to integrated Pinecone

```env
PINECONE_INTEGRATED=true
```

Recreate index as integrated in Pinecone console. Remove OpenAI embedding calls for storage/search (Pinecone embeds server-side).

### Add commands / handlers

Register new handlers in `TelegramBot.register_handlers()`. Keep business logic in methods, not inside handler lambdas.

### Webhook instead of polling

Replace `infinity_polling()` in `TelegramBot.run()` with aiohttp/FastAPI webhook setup — suitable for production behind HTTPS.

### Multi-namespace / multi-tenant

Pass `namespace=f"user_{user_id}"` into `PineconeManager()` for hard isolation instead of metadata filters.

### Plug in tools (function calling)

In `_generate_reply()`, add OpenAI `tools` parameter and route tool calls before saving memory.

### Reduce API costs

- Lower `MEMORY_TOP_K`
- Cache embeddings for identical text
- Batch Pinecone upserts in `pinecone_manager.py`

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | Bot token from BotFather |
| `OPENAI_API_KEY` | — | OpenAI (or compatible proxy) key |
| `OPENAI_BASE_URL` | OpenAI | API base URL |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `PINECONE_API_KEY` | — | Pinecone API key |
| `PINECONE_INDEX_NAME` | — | Index name |
| `PINECONE_INTEGRATED` | `false` | Integrated vs standard index |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (standard mode) |
| `MEMORY_TOP_K` | `8` | Messages injected into LLM context |
| `SYSTEM_PROMPT` | built-in | Assistant system prompt |

---

## Code health check

Static review (no runtime secrets required):

| Check | Status |
|---|---|
| Syntax / imports | OK |
| Async Pinecone client (`await index()`) | OK |
| Namespace `__default__` for Starter indexes | OK |
| First-message detection (search, not stats filter) | OK |
| User-only vector storage (no bot replies) | OK |

### Known limitations

1. **Three embedding calls per message** in standard mode (first-message check, save, context retrieval) — optimize if traffic grows.
2. **Polling only** — no built-in webhook/Docker/health endpoint.
3. **`;nickname` in `first_name`** — if the user's Telegram display name starts with `;`, it is stored as-is in metadata.
4. **Similar messages merge** — cosine threshold `0.85` updates existing vectors instead of creating new ones (`action: updated`).
5. **Integrated mode** — `query_by_text` / `remember_text` require Pinecone integrated inference; standard mode needs OpenAI embeddings.

---

## Development

```bash
# Run bot
python telegram_bot.py

# Probe Pinecone search
python pinecone_manager.py

# Clear user memory from Telegram
/forget
```

Suggested roadmap:

1. Docker + `docker-compose` with env injection
2. GitHub Actions CI (`compileall`, optional lint)
3. Structured logging (JSON) + request tracing
4. Admin dashboard for Pinecone record inspection
5. Rate limiting per `user_id`

---

## License

MIT — use freely, modify, deploy.
