# NodeVision Telegram Bot

Reads trading signals from Redis Streams (published by Go scenario runner) and posts
them to per-author Telegram channels.

## Architecture

```
Go backend (scenario runner)
    │
    ├─ INSERT INTO alerts (postgres)            ← audit log + history
    └─ XADD signals:btc:4h * ...                ← real-time bus
                                                       │
                                                       ▼
              ┌────────────────────────────────────────────────┐
              │ Python bot (this service)                      │
              │   XREADGROUP signals:btc:4h tg_bot_v1 bot_main │
              │      ↓                                         │
              │   aiogram.Bot.send_message(channel_id, text)  │
              │      ↓                                         │
              │   XACK signals:btc:4h tg_bot_v1 <msg-id>      │
              └────────────────────────────────────────────────┘
                                                       │
                                                       ▼
                                            Telegram channel
                                       (subscribers see signal)
```

## Why Redis Streams (not webhook / polling)

- **Persistent** — if bot is offline, messages buffer in Redis
- **Acknowledged** — bot confirms each delivery; unacked messages re-deliver after timeout
- **Replay** — can rewind and reprocess history
- **Multi-consumer** — add per-author bots later without changing Go code

## Run

```bash
# 1. Install
cd telegram-bot
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Configure
cp .env.example .env
# edit .env: TELEGRAM_BOT_TOKEN, TELEGRAM_DEFAULT_CHAT_ID, REDIS_URL, POSTGRES_URL

# 3. Run
python -m src.main
```

## Get TELEGRAM_DEFAULT_CHAT_ID

Telegram channels have negative IDs like `-1001234567890`.

1. Add `@userinfobot` to your channel temporarily (as admin or member)
2. It posts the channel ID
3. Copy the negative number into `.env`
4. Remove `@userinfobot` from the channel

## Test signal injection (without waiting for real model)

```bash
redis-cli XADD signals:btc:4h '*' \
  symbol BTCUSDT \
  interval 4h \
  direction sell \
  entry 76470.62 \
  stop_loss 77517.63 \
  take_profit 74725.59 \
  confidence 0.6065 \
  label trend_aligned \
  bar_time 1745236979
```

The bot should post a formatted message in your channel within seconds.

## Author routing (future)

Currently all signals go to `TELEGRAM_DEFAULT_CHAT_ID`. To route per-author:

```env
TELEGRAM_CHAT_ID_ANONYMOUS=-1001111111111
TELEGRAM_CHAT_ID_SHAOWEI=-1002222222222
TELEGRAM_CHAT_ID_BLONDE=-1003333333333
```

And add an `author` field to the Redis Stream entries.
