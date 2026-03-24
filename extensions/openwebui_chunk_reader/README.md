# Open WebUI Chunk Reader

This folder now contains a real Open WebUI `Action` Function rather than just a separate app launcher.

The installable function is:

```text
extensions/openwebui_chunk_reader/openwebui_action.py
```

## What it does

- Runs as a native Open WebUI Action button under chat messages.
- Opens a rich embedded UI directly inside the conversation.
- Lets you upload a `.txt` / `.md` file or paste text.
- Splits the document into a configurable chunk size.
- Provides transport-style navigation:
  - first
  - previous
  - next
  - last
  - scrubber slider
- Sends the active chunk to a chat-completions endpoint and displays the returned summary.

## Install in Open WebUI

1. Open `Admin Panel` -> `Functions`.
2. Create a new `Action` function.
3. Paste in the contents of [openwebui_action.py](c:/Users/abhij/Code/bringalive/extensions/openwebui_chunk_reader/openwebui_action.py).
4. Save it and enable it globally or assign it to the models you want.
5. Use the new message action button to launch the reader.

## Important Open WebUI note

The embedded UI uses Rich UI embedding. Open WebUI documents that Action Functions can return inline HTML and that these embeds run in a sandboxed iframe. Because of that sandbox, direct calls to the current Open WebUI instance at `/api/chat/completions` work best when:

- `Settings` -> `Interface` -> `Iframe Same-Origin Access` is enabled, and
- the API URL inside the reader is set to `/api/chat/completions`.

If you do not want to enable same-origin iframe access, you can still point the reader at another compatible chat-completions endpoint that accepts cross-origin requests and, if needed, a bearer token.

## Docs used

- Open WebUI plugins overview: https://docs.openwebui.com/features/extensibility/plugin/
- Open WebUI functions: https://docs.openwebui.com/features/extensibility/plugin/functions/
- Open WebUI action functions: https://docs.openwebui.com/features/extensibility/plugin/functions/action/
- Open WebUI rich UI embedding: https://docs.openwebui.com/features/extensibility/plugin/development/rich-ui/
