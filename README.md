# Voice Note Summarization CLI

> Transcribe MP3 voice notes and generate comprehensive meeting summaries with multilingual support (Hindi/English/Hinglish) using local AI models—fully offline with Whisper and Ollama.

A Python command-line tool that transcribes MP3 voice notes, normalizes multilingual content (Hindi, English, Hinglish) to clear English, and generates comprehensive meeting summaries using local AI models. All processing happens entirely offline using OpenAI Whisper for speech-to-text and Ollama with llama3 for language processing and summarization.

## Features

- **Local Speech-to-Text**: Uses OpenAI Whisper (runs entirely on your machine)
- **Multilingual Support**: Handles Hindi, English, and Hinglish content
- **Language Normalization**: Converts mixed-language transcripts to clear, grammatically correct English
- **Comprehensive Summaries**: Generates detailed meeting summaries with:
  - Detailed description of the call context
  - Thorough summary of key points and decisions
  - Chronological minutes of meeting
  - In-depth hierarchical outline of all topics discussed
  - Comprehensive list of facts and data points
  - Complete actionable items with owners and deadlines
- **Multiple Output Formats**: Generates JSON, Markdown, and transcript files
- **Fully Offline**: No cloud services or API keys required

## Requirements

### System Dependencies

- **Python 3.7+** (Python 3.11+ recommended)
- **ffmpeg**: Required by Whisper for audio processing
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg` (Debian/Ubuntu) or use your package manager
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Ollama Setup

1. **Install Ollama**: Download and install from [ollama.ai](https://ollama.ai)
2. **Start Ollama service**: Run `ollama serve` in a terminal (keep it running)
3. **Pull llama3 model**: Run `ollama pull llama3` (this will download the model, ~4.7GB)

### Python Dependencies

Install the required Python packages:

```bash
pip install openai-whisper ollama python-dotenv
```

**Note**: The first time you run Whisper, it will download the model you specify (default: "small" ~500MB). Models are cached locally.

## Installation

1. Clone or download this repository
2. Install system dependencies (ffmpeg, Ollama)
3. Install Python dependencies:
   ```bash
   pip install openai-whisper ollama python-dotenv
   ```
4. Ensure Ollama is running: `ollama serve`
5. Pull the llama3 model: `ollama pull llama3`

## Usage

### Basic Usage

```bash
python summarize_voice.py /path/to/your/audio.mp3
```

### Advanced Options

```bash
python summarize_voice.py /path/to/your/audio.mp3 \
  --whisper-model medium \
  --ollama-model llama3 \
  --outdir ./output
```

### Command-Line Arguments

- `audio` (required): Path to the input MP3 file
- `--whisper-model` (optional): Whisper model size
  - Options: `tiny`, `base`, `small` (default), `medium`, `large`
  - Larger models = better accuracy but slower processing
  - `tiny`: ~39M params, fastest
  - `small`: ~244M params, good balance (default)
  - `medium`: ~769M params, better accuracy
  - `large`: ~1550M params, best accuracy, slowest
- `--ollama-model` (optional): Ollama model name (default: `llama3`)
- `--outdir` (optional): Output directory (default: current directory)

## Output Files

The tool generates three files with timestamps:

1. **`{filename}.{timestamp}.summary.json`**: Structured summary in JSON format
2. **`{filename}.{timestamp}.summary.md`**: Human-readable Markdown summary
3. **`{filename}.{timestamp}.transcript.txt`**: Raw and normalized transcripts

## Example

```bash
# Process a voice note
python summarize_voice.py meeting_recording.mp3 --outdir ./summaries


## License

This project is provided as-is for personal and commercial use.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Notes

- Processing time depends on audio length and selected models
- First run downloads models, subsequent runs are faster
- All models and data remain on your local machine
- No internet connection required after initial setup (except for downloading models)

