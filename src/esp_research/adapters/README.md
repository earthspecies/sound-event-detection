# Adapters

Two things live here:

1. **`HttpClient`** — thin synchronous HTTP client for POSTing dict payloads to a remote
   inference service. Implements the `InferenceModel` protocol (callable, returns a dict).
2. **Audio utilities** (`esp_research.utils.audio`) — helpers for coercing and encoding
   audio before sending it over HTTP.

---

## HttpClient

### Construction

```python
from esp_research.adapters import HttpClient, HttpClientConfig, HttpAuthConfig

# Direct
client = HttpClient(
    url="http://localhost:8001",
    route="my-route",          # POST goes to {url}/{route}; omit to POST to url
    auth=HttpAuthConfig(header="Authorization", value="Bearer secret"),
    timeout=30.0,
    retries=3,
    audio_key="audio",         # key in payload whose value is raw audio bytes
    audio_format="wav",        # adds {"audio_format": "wav"} to the payload
)

# From a YAML config file
config = HttpClientConfig.from_sources(yaml_file="httpclient_cfg.yml")
client = HttpClient.from_config(config)
```

**`HttpClientConfig` YAML fields:**

```yaml
url: "http://localhost:8001"
route: "my-route"          # optional
timeout: 30.0
retries: 3
audio_key: audio           # optional — triggers base64 encoding of that key
audio_format: wav          # optional — adds audio_format to the payload
auth:                      # optional
  header: Authorization
  value: "Bearer ${MY_TOKEN}"
```

### Calling

```python
response = client({"audio": audio_bytes, "query": "What species is this?"})
# response is the parsed JSON dict the server returned
```

`**kwargs` are merged into the payload dict:

```python
response = client(audio=audio_bytes, query="What species is this?")
```

### Other methods

```python
config_dict = client.describe()  # GET {url}/ — returns server's live config
client.close()                   # close the underlying httpx.Client
```

`HttpClient` is also a context manager:

```python
with HttpClient(url="http://localhost:8001") as client:
    result = client(payload)
```

### Audio encoding

When `audio_key` is set, the client base64-encodes the value at that key before sending.
Accepts `bytes`, `list[float]`, or `np.ndarray` (floating dtype). Batches (lists of samples)
are encoded element-wise. If `audio_format` is also set, `{"audio_format": "<fmt>"}` is
added to the payload.

---

## Audio utilities (`esp_research.utils.audio`)

### `coerce_audio_bytes(audio_data, audio_key) → bytes`

Convert audio to raw bytes.

| Input type | Output |
|---|---|
| `bytes` | returned as-is |
| `list[float]` | cast to `float32`, `.tobytes()` |
| `np.ndarray` (float) | cast to `float32`, `.tobytes()` |

Raises `TypeError` for non-float arrays or unsupported types; `ValueError` for empty input.

```python
from esp_research.utils.audio import coerce_audio_bytes

raw = coerce_audio_bytes(np.array([0.1, -0.2, 0.0], dtype=np.float32), "audio")
```

### `float_array_to_wav(audio_data, audio_key, sample_rate) → bytes`

Convert float samples (range −1.0 to 1.0) to WAV bytes (mono, 16-bit PCM).
If the input is already `bytes`, it is returned unchanged.

```python
from esp_research.utils.audio import float_array_to_wav

wav_bytes = float_array_to_wav(audio_array, "audio", sample_rate=16000)
```

### `bytes_to_base64_string(data) → str`

Base64-encode bytes to a UTF-8 string.

```python
from esp_research.utils.audio import bytes_to_base64_string

encoded = bytes_to_base64_string(raw_bytes)
```
