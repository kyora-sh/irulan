# kyora irulan API (spec)

The HTTP API the `irulan` client expects. kyora-platform implements it.

| part | status |
|---|---|
| Inference: `/v1/chat/completions`, `/v1/models` | **planned**: no hosted endpoint yet |
| Datasets: `/v1/datasets...` | **planned** |
| Fine-tunes: `/v1/fine_tunes...` | **planned** |

Until a hosted base URL exists, the client has no default. Every call raises
`irulan.NotAvailableYet` (CLI exit code 3). It works against any server that
implements this spec when `IRULAN_BASE_URL` points at it. The client also
raises `NotAvailableYet` when a server answers `501 Not Implemented`, so an
implementation can ship the parts in stages.

The words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.

## Conventions

- **Base URL.** It includes the version, for example
  `https://<host>/v1`. All paths below are relative to `/v1`.
- **Auth.** `Authorization: Bearer <key>` on every request. A missing or
  invalid key gets a `401`. A key belongs to one user. The server MUST scope
  every dataset, fine-tune and private model to that user.
- **Bodies.** JSON (`application/json`, UTF-8), except the dataset upload.
- **Timestamps.** Unix seconds (`created_at`, `finished_at`).
- **IDs.** Opaque strings. The client never parses them.
- **Idempotency.** The client sends `Idempotency-Key: <uuid>` on every POST
  and reuses it when it retries. The server SHOULD return the original
  response for a repeated key (24 h window), so a retried create never makes
  two fine-tunes.
- **Lists.** `GET` on a collection returns
  `{"object": "list", "data": [...], "has_more": bool}`, newest first. It
  takes `?limit=` (default 20, max 100) and `?after=<id>` for paging.
- **Retries.** The client retries `408`, `429`, `500`, `502`, `503` and
  `504` twice, honouring `Retry-After` (seconds). It does not retry other
  `4xx` responses or `501`.

### Errors

Every error uses the OpenAI-style shape:

```json
{"error": {"message": "dataset ds_123 is scrubbed, not ready", "type": "invalid_request_error",
           "code": "dataset_not_ready", "param": "dataset_id"}}
```

| status | `type` | used for |
|---|---|---|
| 400 | `invalid_request_error` | bad JSON, an invalid dataset file (`code`: `invalid_jsonl`, `invalid_record`, `duplicate_id`, `too_large`), a bad name |
| 401 | `authentication_error` | missing or invalid key |
| 403 | `permission_error` | key not allowed (e.g. fine-tuning not enabled for the account) |
| 404 | `invalid_request_error` | unknown id or model (`dataset_not_found`, `fine_tune_not_found`, `model_not_found`) |
| 409 | `invalid_request_error` | wrong state (`dataset_not_ready`, `dataset_not_scrubbed`, `fine_tune_finished`, `name_taken`) |
| 413 | `invalid_request_error` | upload over the size limit |
| 429 | `rate_limit_error` | rate or quota limit; send `Retry-After` |
| 5xx | `server_error` | server fault |
| 501 | `not_implemented` | the endpoint is not live yet |

## Inference (planned)

### `POST /chat/completions`

OpenAI-compatible. The client sends:

```json
{
  "model": "irulan-swe",
  "messages": [
    {"role": "system", "content": "Rewrite the text in the writer's own voice: terse, direct, short lines. Keep the meaning, asks, facts and numbers. Return only the rewrite."},
    {"role": "user", "content": "I wanted to follow up and ask whether you might have a moment to review the PR."}
  ],
  "temperature": 0.3,
  "max_tokens": 512
}
```

- `model`: `irulan-swe` (the shared base) or one of the caller's own
  `irulan-<user>-<name>` models. Another user's model is a `404`.
- The system message is the versioned instruction the training data is
  built with (`system_v1`, shown above; `irulan.SYSTEM_PROMPT` in the
  client). The platform's data prep MUST put exactly this text into training
  data for a model that reports `system_version: "system_v1"`. A new
  instruction gets a new version and never replaces the text of an old one.
- If a request has no system message, the server SHOULD insert the one for
  the model's `system_version`.
- `temperature`, `max_tokens`, `top_p`, `stop`, `n` and `stream` behave as
  in OpenAI's API. The client sends only what the caller sets, and does not
  stream.

The response follows OpenAI's `chat.completion` object. The client reads
`choices[0].message.content`.

### `GET /models`

```json
{"object": "list", "data": [
  {"id": "irulan-swe", "object": "model", "owned_by": "kyora", "created": 1760000000,
   "system_version": "system_v1"},
  {"id": "irulan-jdoe-voice", "object": "model", "owned_by": "jdoe", "created": 1760100000,
   "system_version": "system_v1", "fine_tune_id": "ft_abc", "base_model": "irulan-swe"}
]}
```

It lists `irulan-swe` and the caller's own succeeded fine-tunes.

### Model ids

- `irulan-swe` is the shared base model.
- `irulan-<user>-<name>` is a user's fine-tune. `<user>` is the account's
  handle and `<name>` is the `name` given at creation. Both are lowercase
  `[a-z0-9-]`, 1 to 40 characters each. A name is unique per user and is
  freed when its fine-tune is deleted.

## Datasets (planned)

A dataset is the **raw extract** written by `irulan extract`. The
`irulan.raw.v1` upload format is JSONL with one record per line:

```json
{"id": "3f2a9c0d1e4b5a68", "text": "can u rq bump the worker to 2 gpus?",
 "meta": {"source": "claude-code", "session": "9b1c0e7f2a3d4c5b", "timestamp": "2026-01-01T10:00:00Z",
          "words": 9, "lines": 1}}
```

- `id`: string, unique within the file.
- `text`: non-empty string. This is the writer's own typed text, and it is
  **not scrubbed**. It may contain names, hosts, URLs, paths, IDs and
  secrets. If the user passed `--redact-secrets-locally`, obvious
  credentials are already replaced with `<SECRET>`.
- `meta.session` is an opaque hash of the local project and session id. The
  platform uses it to split train, val and test by session. The local
  project directory name, which contains the user's home path, is not sent.
- Other `meta` keys are optional. The server MUST ignore keys it does not
  know.

### Handwritten pairs (`chat.v1`)

The pairs `irulan studio` writes are uploaded with `irulan upload --kind
pairs` as `chat.v1` records for kyora's `sft` recipe, one conversation per
pair:

```json
{"id": "hw_3f2a9c0d1e4b5a68",
 "messages": [{"role": "system", "content": "Rewrite the text in the writer's own voice: terse, direct, short lines. Keep the meaning, asks, facts and numbers. Return only the rewrite."},
              {"role": "user", "content": "I wanted to follow up and ask whether you might have a moment to review the PR."},
              {"role": "assistant", "content": "any chance u can look at the pr today?"}],
 "meta": {"source": "handwritten", "session": "handwritten-2026-01-01", "timestamp": "2026-01-01T10:00:00Z"}}
```

- `id`: the pair's id, unique within the file.
- `messages`: the `system_v1` instruction, the assistant-style text as the
  user turn and the writer's own version as the assistant turn. Every
  `content` is a non-empty string, and the last turn is the assistant's.
  The text is **not scrubbed** either.
- `meta.session` groups the pairs written on one day, so a day's pairs land
  in one split.

The `sft` recipe has no back-translation, so its stages are `uploaded`,
`scrubbed` (then the review gate when `review` is `required`) and `ready`.

### `POST /datasets`

The request is `multipart/form-data` with these parts:

| part | value |
|---|---|
| `file` | the JSONL, `Content-Type: application/jsonl` |
| `format` | `irulan.raw.v1`, or `chat.v1` for handwritten pairs |
| `name` | display name |
| `purpose` | `fine-tune` |
| `review` | `required` (default) or `skip` |
| `recipe` | optional; the client sends `sft` with `chat.v1` |

The server validates the whole file before it accepts it: UTF-8 JSONL,
every line an object with a string `id` and a non-empty string `text` (or,
for `chat.v1`, valid `messages`), ids
unique, and size within the limit (SHOULD be at least 100 MB). On an
invalid file it answers `400` with the 1-based line number in `message` and
does not keep the file.

For files over the multipart limit, the server MAY offer a signed-URL
upload:

1. `POST /datasets/uploads {"name", "format", "bytes", "review"}` →
   `{"dataset_id", "upload_url", "expires_at"}`.
2. `PUT <upload_url>` with the raw JSONL.
3. `POST /datasets/{id}/complete` → the dataset object.

The current client uses multipart only.

The response is a dataset object:

```json
{"id": "ds_123", "object": "dataset", "name": "cc_messages.jsonl", "format": "irulan.raw.v1",
 "status": "uploaded", "records": 1834, "bytes": 412004, "created_at": 1760000000,
 "review": {"required": true, "status": "pending", "candidates": 0},
 "stages": {"uploaded": 1760000000, "scrubbed": null, "pairs": null, "ready": null},
 "stats": {}, "error": null}
```

### Processing stages

The server does all of the following. The client does none of it.

| `status` | what has happened |
|---|---|
| `uploaded` | The file was received and validated, and scrubbing is running. |
| `scrubbed` | Secrets, URLs, emails, IPs, hosts, paths, IDs and detected names were replaced with typed placeholders (`<SECRET>`, `<URL>`, `<EMAIL>`, `<IP>`, `<HOST>`, `<PATH>`, `<ID>`, `<NAME>`). Messages with no prose left were dropped and the rest deduplicated. The name review list is ready. With `review.required`, processing waits here for `POST /datasets/{id}/review`. |
| `pairs` | Back-translation is done: each scrubbed message is a target, a model has written the assistant-style input for it, and bad pairs were filtered out. |
| `ready` | The training set is built: placeholders were replaced with invented surrogates and the data was split by session into train, val and test in chat format with the `system_v1` instruction. The dataset can now be fine-tuned. |
| `failed` | `error` says why (`{"code", "message"}`). |

`stages` holds the time each stage was reached. `stats` holds counts only,
never text: messages kept and dropped per reason, placeholders per type,
pairs per split.

### `GET /datasets/{id}`

Returns the dataset object. The client's `datasets.wait(id, until=stage)`
polls this until the dataset reaches `until`, and raises if it is `failed`.

### `GET /datasets`

Returns a list of dataset objects.

### `GET /datasets/{id}/review`

The platform's name and entity review list. These are tokens in the
scrubbed text that may be names of people, customers, companies, products
or machines and were not already replaced. The user reads the list before
pairs are generated, so a missed name never reaches training.

```json
{"object": "dataset.review", "dataset_id": "ds_123", "status": "pending",
 "candidates": [{"term": "Zorblax", "count": 14, "example": "ask Zorblax whether <URL> is live"}]}
```

`example` is cut from the **scrubbed** text, at most 80 characters. Before
the dataset reaches `scrubbed`, this endpoint answers `409 dataset_not_scrubbed`.

### `POST /datasets/{id}/review`

`{"approve": true, "denylist": ["Zorblax", "Jane Example"]}`

The listed terms are scrubbed as `<NAME>` (case-insensitive, whole word),
and a term followed by `-<digits>` is scrubbed as `<HOST>`. The server
rescrubs the stored scrubbed dataset with them and then goes on to `pairs`.
The response is the dataset object. `approve: false` stores the terms
without moving on.

### `DELETE /datasets/{id}`

Deletes the dataset and everything derived from it: the scrubbed records,
the review list, the pairs and the training splits. Fine-tuned models
trained on it are not deleted; delete those separately. The response is
`{"id", "object": "dataset", "deleted": true}`. The client's
`irulan datasets delete` asks first.

## Fine-tunes (planned)

### `POST /fine_tunes`

```json
{"dataset_id": "ds_123", "base_model": "irulan-swe", "name": "voice",
 "hyperparameters": {"epochs": 2, "learning_rate_multiplier": 1.0, "lora_rank": 16}}
```

- `dataset_id` MUST be `ready`. Otherwise the server answers
  `409 dataset_not_ready`.
- `base_model` defaults to `irulan-swe`.
- `name` is `[a-z0-9-]`, 1 to 40 characters, and unique per user
  (`409 name_taken`).
- `hyperparameters` is optional. The server picks defaults, rejects
  unknown keys with a `400`, and echoes back the effective values.

It trains a LoRA adapter on the base model. The response is a fine-tune
object.

### Fine-tune object

```json
{"id": "ft_abc", "object": "fine_tune", "dataset_id": "ds_123", "base_model": "irulan-swe",
 "name": "voice", "status": "running", "created_at": 1760000000, "started_at": 1760000300,
 "finished_at": null, "hyperparameters": {"epochs": 2, "learning_rate_multiplier": 1.0, "lora_rank": 16},
 "fine_tuned_model": null,
 "metrics": {"step": 120, "total_steps": 400, "train_loss": 1.42, "val_loss": 1.51},
 "events": [{"created_at": 1760000300, "level": "info", "message": "training started"}],
 "error": null}
```

| `status` | meaning |
|---|---|
| `queued` | waiting for capacity |
| `running` | training; `metrics` updates |
| `succeeded` | `fine_tuned_model` is set (`irulan-<user>-<name>`) and usable at `/chat/completions` |
| `failed` | `error` is `{"code", "message"}` |
| `cancelled` | cancelled by the user |

`events` is append-only and holds the latest 100 events. The server MAY
also expose all of them at `GET /fine_tunes/{id}/events` (a list).
`metrics` holds the latest values, and when the job ends it holds the final
eval numbers.

### `GET /fine_tunes/{id}`, `GET /fine_tunes`

Return the fine-tune object and a list of them. `finetunes.wait(id)` polls
until a terminal status. It raises `irulan.FineTuneFailed` for `failed` and
`cancelled`.

### `POST /fine_tunes/{id}/cancel`

Cancels a `queued` or `running` job and returns the object with
`status: "cancelled"`. On a finished job it answers `409 fine_tune_finished`.

## Data handling

Uploads hold unscrubbed text, so these rules are part of the spec:

- The platform MUST scrub every dataset on receipt, before it stores
  anything.
- It MUST persist only the scrubbed dataset, encrypted at rest.
- It MUST NOT retain the original (unscrubbed) upload.
- `DELETE /datasets/{id}` deletes the dataset and everything derived from
  it (see above).
- Dataset and fine-tune objects, `stats`, events and errors MUST NOT
  contain message text. The only exception is the review list's `example`
  snippets, which are cut from the scrubbed text.

The client states these rules in `irulan upload`'s confirmation text and in
the README.
