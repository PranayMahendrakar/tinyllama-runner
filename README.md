# TinyLlama Chat

A browser-based chat interface for [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) that runs entirely on **GitHub Actions** — no servers, no local setup.

**Live Chat:** https://pranaymahendrakar.github.io/tinyllama-runner/

## How it works

1. You type a question in the chat UI
2. The frontend triggers a `workflow_dispatch` event on the `chat.yml` workflow via GitHub API
3. GitHub Actions spins up an Ubuntu runner, loads TinyLlama (1.1B params, CPU), and generates a response
4. The answer is written to `answers/{msg_id}.json` and committed back to the repo
5. The frontend polls `raw.githubusercontent.com/answers/{msg_id}.json` until the answer appears (~2-3 min)

## Setup (one-time, 5 minutes)

The chat requires a **GitHub Fine-Grained Personal Access Token** with only `Actions: write` permission on this repo. This token is needed to trigger the `workflow_dispatch` event.

### Step 1: Create the token

1. Go to: https://github.com/settings/personal-access-tokens/new
2. Set **Token name**: `tinyllama-chat`
3. Set **Expiration**: 90 days (or longer)
4. Under **Repository access**, choose **Only select repositories** → select `tinyllama-runner`
5. Under **Permissions → Repository permissions**, set:
   - `Actions`: **Read and write**
   - Everything else: No access
6. Click **Generate token** and copy it

### Step 2: Use the chat

1. Open https://pranaymahendrakar.github.io/tinyllama-runner/
2. A setup box will appear asking for your token
3. Paste the token and click **Save & start chatting**
4. The token is stored only in your browser's `localStorage` — never sent anywhere except GitHub's API

> The token can only trigger workflows on this one repo. It cannot read private data, delete code, or access anything else.

## Files

- `app.py` — original TinyLlama inference script
- `index.html` — GitHub Pages chat UI
- `.github/workflows/chat.yml` — workflow triggered per message, writes answer to `answers/`
- `.github/workflows/run.yml` — original run workflow
- `requirements.txt` — Python dependencies
- `answers/` — response files written by Actions and read by the frontend
