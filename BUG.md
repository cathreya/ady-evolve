# Bug report — fork inference rejects the model UUID (only the slug works)

**Captured:** 2026-06-21 ~14:42 UTC
**Endpoint:** `https://inference.beta.hud.ai/v1/chat/completions`
**Account/team:** cathreya98@gmail.com
**Affected fork:** `tempora-gptoss20b` (UUID `dff37ae9-89ca-4560-ae6e-8a3040e17b9e`), forked from `openai/gpt-oss-20b`

## Summary

Inference on a team-owned fork **fails when the model is addressed by its UUID**
(`400 invalid_request_error: "model '<uuid>' is not supported"`), but **succeeds
when addressed by its slug** (`tempora-gptoss20b`). The base model by name also
works. This is a behavior change: **the same fork trained successfully by UUID
for ~46 steps earlier (with normal inference rollouts), then UUID inference began
returning 400 at roughly 03:00 PDT on 2026-06-21.** The training/`head` endpoints
still resolve the fork by UUID — only the inference gateway rejects the UUID.

## Reproduction (authenticated)

Same request body to each, only `model` differs:

| model | result | x-request-id |
|---|---|---|
| `dff37ae9-89ca-4560-ae6e-8a3040e17b9e` (fork UUID) | **HTTP 400** | `5fca009d-68a1-4a08-99f8-680f399de502` |
| `tempora-gptoss20b` (fork slug) | HTTP 200 OK | `8705bf31-f070-4bb8-b6e1-459e19a9a796` |
| `openai/gpt-oss-20b` (base by name) | HTTP 200 OK | `639b94e6-c95e-4e8a-9852-187a607b152c` |

### Failing call
```bash
curl -i -X POST https://inference.beta.hud.ai/v1/chat/completions \
  -H "Authorization: Bearer $HUD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"dff37ae9-89ca-4560-ae6e-8a3040e17b9e","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```
Response:
```
HTTP/2 400
{"error":{"message":"model 'dff37ae9-89ca-4560-ae6e-8a3040e17b9e' is not supported","type":"invalid_request_error","param":null}}
```

### Working call (slug)
```bash
curl -i -X POST https://inference.beta.hud.ai/v1/chat/completions \
  -H "Authorization: Bearer $HUD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"tempora-gptoss20b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```
→ `HTTP 200`, normal completion.

## Scope

Reproduced on **all team forks by UUID** (gpt-oss-20b, Nemotron-Super-120B, and
three Qwen3-8B forks) — every one returns the same `400 "... is not supported"`.
Both base models (`openai/gpt-oss-20b`, `Qwen/Qwen3-8B`) serve fine by name.
So it is not weight- or family-specific; it is **UUID→fork resolution on the
inference gateway**.

## Impact

- Blocked all RL training/eval for ~3+ hours while we (wrongly) assumed a total
  fork-serving outage and retried by UUID.
- Workaround found: **address forks by slug, not UUID.** Training endpoints
  (`TrainingClient`, `.head()`) still accept the UUID, which is the inconsistency.

## Questions / requests

1. Is the UUID→fork inference path intentionally deprecated, or did it regress?
   If intentional, the `400` message should say "use the model slug" rather than
   "not supported", and ideally the inference gateway should accept the UUID like
   the training endpoints do (consistency).
2. Was there a gateway deploy around 03:00 PDT 2026-06-21 that changed the
   accepted inference identifier? That matches when UUID inference started failing.

Auth note: `$HUD_API_KEY` must be exported in the shell for the curls above
(it lives in `~/.hud/.env`, which the `hud` CLI auto-loads but a bare curl does
not — an unset key returns `401 unauthorized: No API key provided`).
