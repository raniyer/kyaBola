#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import whisper

try:
    import ollama
except Exception as exc:  # pragma: no cover
    ollama = None


def transcribe_audio(audio_path: Path, whisper_model: str = "small") -> str:
    """
    Transcribe an audio file to text using OpenAI Whisper (local).
    """
    model = whisper.load_model(whisper_model)
    # fp16 False for broader compatibility (CPU or non-CUDA)
    result = model.transcribe(str(audio_path), fp16=False)
    return result.get("text", "").strip()


def ensure_ollama_or_exit(model_name: str = "llama3") -> None:
    """
    Ensure Ollama is importable and the target model exists/loads.
    """
    if ollama is None:
        print("Error: 'ollama' package not available. Install with: pip install ollama", file=sys.stderr)
        sys.exit(1)
    try:
        # This call will raise if the daemon isn't running
        _ = ollama.list()
    except Exception as exc:
        print("Error: Ollama daemon not reachable. Start it with: 'ollama serve'", file=sys.stderr)
        sys.exit(1)
    # Try to pull/ensure the model is present
    try:
        ollama.show(model_name)
    except Exception:
        # Attempt to pull if not present
        print(f"Pulling model '{model_name}' via Ollama...", file=sys.stderr)
        ollama.pull(model_name)


def ensure_ffmpeg_or_exit() -> None:
    """
    Ensure ffmpeg is available on PATH (whisper dependency).
    """
    if shutil.which("ffmpeg") is None:
        msg = (
            "Error: 'ffmpeg' not found on PATH. Install it, e.g. on macOS:\n"
            "  brew install ffmpeg\n"
            "Then re-run this command."
        )
        print(msg, file=sys.stderr)
        sys.exit(1)


def call_ollama_json(model: str, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> dict:
    """
    Call Ollama chat with a system + user prompt, requesting strict JSON output.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    content = response.get("message", {}).get("content", "").strip()
    # Try to locate JSON in response
    try:
        # If model returned markdown code fences, strip them
        cleaned = content
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`\n ")
            # Remove language tag if present (e.g., json)
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1 :].strip()
        data = json.loads(cleaned)
        return data
    except Exception:
        # Last resort: try to extract first/last brace
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except Exception:
                pass
        raise RuntimeError("Failed to parse JSON from Ollama response.\nResponse was:\n" + content)


def normalize_transcript_to_english(model: str, transcript: str) -> str:
    """
    Use llama3 to convert Hindi/English/Hinglish into clear, well-formed English text.
    """
    system_prompt = (
        "You are a meticulous language normalizer. Convert any Hindi, English, or Hinglish speech "
        "into fluent, concise, and grammatically correct English while preserving meaning. Do not invent facts."
    )
    user_prompt = (
        "Convert the following raw transcript into clear English sentences. Keep speaker-agnostic narration.\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Return only the normalized text (no preface, no explanation)."
    )
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.2},
    )
    return response.get("message", {}).get("content", "").strip()


def summarize_structured(model: str, normalized_text: str) -> dict:
    """
    Ask llama3 to produce a strict JSON with the required sections.
    """
    system_prompt = (
        "You are an expert meeting summarizer. Read the conversation thoroughly and produce a comprehensive, detailed, factual, and actionable summary. "
        "Extract all relevant information, context, nuances, and details from the transcript. "
        "Do not add information not present in the transcript, but be thorough in capturing everything that was discussed."
    )
    user_prompt = (
        "Analyze the following conversation in detail and produce STRICT JSON with this schema:\n\n"
        "{\n"
        "  \"description\": string,\n"
        "  \"summary\": string,\n"
        "  \"minutes_of_meeting\": string,\n"
        "  \"detailed_outline\": string,\n"
        "  \"facts\": string[],\n"
        "  \"actionable_items\": string[]\n"
        "}\n\n"
        "Detailed Guidelines (be thorough and comprehensive in ALL sections):\n\n"
        "- description: Provide a detailed, comprehensive description of the call purpose, context, participants (if mentioned), and overall scope. Include background information, meeting type, and any relevant context that sets the stage. Should be 5-10 sentences, not just 1-3.\n\n"
        "- summary: Create a thorough, detailed summary covering all key points, decisions, discussions, agreements, disagreements, and outcomes. Include specific details, numbers, dates, names, and any important nuances. Should be 10-20 sentences, covering all major aspects comprehensively.\n\n"
        "- minutes_of_meeting: Write detailed, chronological minutes covering the entire conversation flow. Include who said what (if speakers are identifiable), when topics were discussed, the progression of ideas, detailed discussions, questions raised, answers given, and the sequence of events. Be comprehensive and capture the full narrative arc of the meeting.\n\n"
        "- detailed_outline: Create a comprehensive, in-depth hierarchical outline of EVERYTHING discussed. Include all topics, subtopics, sub-subtopics, discussions, decisions, key points, arguments, examples, data points, and details. Use clear hierarchical structure with sections, subsections, and bullet points. This should be extremely detailed and serve as a complete reference of the meeting content.\n\n"
        "- facts: Extract ALL objective facts, statements, data points, numbers, dates, names, locations, and verifiable information mentioned in the conversation. Be comprehensive - include every factual statement, statistic, reference, and concrete piece of information. This should be an extensive list.\n\n"
        "- actionable_items: Extract ALL tasks, action items, commitments, deadlines, responsibilities, and follow-ups mentioned. Include who is responsible (if mentioned), deadlines, specific deliverables, and any conditions or dependencies. Be thorough and capture every actionable item discussed.\n\n"
        "Remember: Be comprehensive, detailed, and thorough in EVERY section. Capture as much information as possible from the transcript.\n\n"
        f"Conversation:\n{normalized_text}\n\n"
        "Return ONLY the JSON."
    )
    return call_ollama_json(model=model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.3)


def to_markdown(data: dict) -> str:
    def section(title: str, body: str) -> str:
        return f"### {title}\n\n{body.strip()}\n\n"

    def bullets(title: str, items: list) -> str:
        lines = [f"### {title}", ""]
        for it in items:
            lines.append(f"- {it}")
        lines.append("")
        return "\n".join(lines)

    md = ["# Call Summary", ""]
    md.append(section("Description", data.get("description", "")))
    md.append(section("Summary", data.get("summary", "")))
    md.append(section("Minutes of Meeting", data.get("minutes_of_meeting", "")))
    md.append(section("Detailed Outline", data.get("detailed_outline", "")))
    md.append(bullets("Facts Spoken", data.get("facts", [])))
    md.append(bullets("Actionable Items", data.get("actionable_items", [])))
    return "\n".join(md).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice note summarization CLI (Whisper + Ollama llama3)")
    parser.add_argument("audio", type=str, help="Path to input .mp3 file")
    parser.add_argument("--whisper-model", type=str, default="small", help="Whisper model size (tiny/base/small/medium/large)")
    parser.add_argument("--ollama-model", type=str, default="llama3", help="Ollama model name (e.g., llama3)")
    parser.add_argument("--outdir", type=str, default=".", help="Directory to write outputs")
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        print(f"Input not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ensure_ollama_or_exit(args.ollama_model)
    ensure_ffmpeg_or_exit()

    print("Transcribing audio with Whisper...", file=sys.stderr)
    transcript = transcribe_audio(audio_path, whisper_model=args.whisper_model)
    if not transcript:
        print("Transcription returned empty text.", file=sys.stderr)
        sys.exit(1)

    print("Normalizing language to English with llama3...", file=sys.stderr)
    normalized = normalize_transcript_to_english(args.ollama_model, transcript)

    print("Generating structured summary with llama3...", file=sys.stderr)
    summary_data = summarize_structured(args.ollama_model, normalized)

    # Compose file names
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = audio_path.stem
    json_path = outdir / f"{stem}.{timestamp}.summary.json"
    md_path = outdir / f"{stem}.{timestamp}.summary.md"
    transcript_path = outdir / f"{stem}.{timestamp}.transcript.txt"

    # Save raw transcript
    print(f"Writing: {transcript_path}", file=sys.stderr)
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("=== Raw Transcript (from Whisper) ===\n\n")
        f.write(transcript)
        f.write("\n\n=== Normalized Transcript (English) ===\n\n")
        f.write(normalized)

    print(f"Writing: {json_path}", file=sys.stderr)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print(f"Writing: {md_path}", file=sys.stderr)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(to_markdown(summary_data))

    # Also print JSON to stdout for CLI usage
    print(json.dumps(summary_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



