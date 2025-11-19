#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
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


def clean_control_characters(text: str) -> str:
    """
    Remove or escape invalid control characters from JSON strings.
    JSON only allows \n, \r, and \t as control characters in strings.
    All other control characters (0x00-0x1F except 0x09, 0x0A, 0x0D) are invalid.
    """
    result = []
    for char in text:
        code = ord(char)
        # Allow printable characters (>= 0x20), and valid JSON control chars (\t, \n, \r)
        if code >= 0x20 or code in (0x09, 0x0A, 0x0D):
            result.append(char)
        # Replace other control characters with space or remove them
        # We'll replace with space to preserve structure
        elif code < 0x20:
            result.append(' ')
    return ''.join(result)


def balance_json_structure(text: str) -> str:
    """
    Ensure JSON braces/brackets are balanced by appending missing closing tokens.
    This is helpful when the LLM truncates the response before closing the object/array.
    """
    stack = []
    in_string = False
    escape = False

    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == '\\\\':
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == '{':
            stack.append('}')
        elif char == '[':
            stack.append(']')
        elif char in (']', '}'):
            if stack and stack[-1] == char:
                stack.pop()
            else:
                # Mismatch (extra closing token). Discard to keep parser lenient.
                continue

    if stack:
        text += ''.join(reversed(stack))
    return text


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
    
    if not content:
        raise RuntimeError("Ollama returned empty response. Check if the model is working correctly.")
    
    # Remove BOM if present
    if content.startswith('\ufeff'):
        content = content[1:]
    
    # Clean invalid control characters and balance truncated structures
    content = clean_control_characters(content)
    content = balance_json_structure(content)
    
    # Try multiple strategies to extract JSON
    strategies = [
        # Strategy 1: Direct JSON parse
        ("direct_parse", lambda c: json.loads(c)),
        # Strategy 2: Remove markdown code fences
        ("remove_code_fences", lambda c: json.loads(c.strip("`\n ")) if c.startswith("```") else None),
        # Strategy 3: Remove markdown code fences with language tag
        ("remove_code_fences_with_tag", lambda c: json.loads(c.split("\n", 1)[1].strip("`\n ")) if c.startswith("```") else None),
        # Strategy 4: Extract JSON between first { and last }
        ("extract_braces", lambda c: json.loads(c[c.find("{"):c.rfind("}")+1]) if "{" in c and "}" in c else None),
    ]
    
    last_error = None
    for strategy_name, strategy in strategies:
        try:
            result = strategy(content)
            if result is not None:
                return result
        except json.JSONDecodeError as e:
            last_error = e
            # Only show detailed error for direct parse to avoid spam
            if strategy_name == "direct_parse":
                print(f"\nDEBUG: Direct parse failed: {e}", file=sys.stderr)
                if hasattr(e, 'pos') and e.pos < len(content):
                    start = max(0, e.pos - 100)
                    end = min(len(content), e.pos + 100)
                    print(f"Error at position {e.pos}, context: {content[start:end]}", file=sys.stderr)
            continue
        except (ValueError, AttributeError) as e:
            continue
    
    # If all strategies fail, try to clean and parse more aggressively
    # Remove any text before first { and after last }
    start_idx = content.find("{")
    end_idx = content.rfind("}")
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = content[start_idx:end_idx + 1]
        json_str = balance_json_structure(json_str.strip())
        
        # Try multiple cleaning strategies
        cleaning_attempts = [
            # Attempt 1: Direct parse
            ("direct", json_str),
            # Attempt 2: Remove trailing commas before closing braces/brackets
            ("remove_trailing_commas", re.sub(r',(\s*[}\]])', r'\1', json_str)),
        ]
        
        last_error = None
        for attempt_name, cleaned_json in cleaning_attempts:
            try:
                return json.loads(cleaned_json)
            except json.JSONDecodeError as e:
                last_error = e
                # Show detailed error for debugging
                error_pos = getattr(e, 'pos', None)
                if error_pos is not None and error_pos < len(cleaned_json):
                    start_context = max(0, error_pos - 100)
                    end_context = min(len(cleaned_json), error_pos + 100)
                    context = cleaned_json[start_context:end_context]
                    print(f"\nDEBUG: JSON parse error ({attempt_name}) at position {error_pos}:", file=sys.stderr)
                    print(f"Error message: {e}", file=sys.stderr)
                    print(f"Context around error:\n{context}", file=sys.stderr)
                    print(f"Character at error position: {repr(cleaned_json[error_pos]) if error_pos < len(cleaned_json) else 'N/A'}", file=sys.stderr)
                continue
        
        # If direct parsing failed, try to repair common JSON issues
        try:
            repaired = json_str
            # Clean invalid control characters first
            repaired = clean_control_characters(repaired)
            repaired = balance_json_structure(repaired)
            # Remove null bytes (redundant but safe)
            repaired = repaired.replace('\x00', '')
            
            # Fix pattern: { "key": { "just a string" } } -> { "key": "just a string" }
            # This pattern appears when llama3 creates nested objects with string values
            # Match: { "key": { "value" } } and convert to { "key": "value" }
            def fix_nested_string_objects(text):
                """Fix objects that contain just a string value instead of key-value pairs."""
                # Pattern: "key": { "value" } where value is just a string (may span multiple lines)
                # We need to handle multi-line cases and apply iteratively for nested structures
                import re
                # Match: "key": { "value" } - handle multi-line with DOTALL flag
                pattern = r'"([^"]+)":\s*\{\s*"([^"]+)"\s*\}'
                
                def replace_func(match):
                    key = match.group(1)
                    value = match.group(2)
                    # Escape quotes and newlines in value
                    value_escaped = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
                    return f'"{key}": "{value_escaped}"'
                
                # Apply the fix multiple times to handle nested cases
                result = text
                max_iterations = 10
                for _ in range(max_iterations):
                    new_result = re.sub(pattern, replace_func, result, flags=re.MULTILINE | re.DOTALL)
                    if new_result == result:
                        break
                    result = new_result
                
                return result
            
            repaired = fix_nested_string_objects(repaired)
            
            # Try parsing the repaired JSON
            return json.loads(repaired)
        except (json.JSONDecodeError, Exception) as e:
            # If repair didn't work, try one more approach: use a JSON repair library pattern
            # Convert the detailed_outline to a string if it's malformed
            try:
                # Try to parse and fix the structure programmatically
                # First, try to extract just the parts that are valid
                data = {}
                # Use a more lenient approach - try to extract each field separately
                for field in ["description", "summary", "minutes_of_meeting", "detailed_outline", "facts", "actionable_items"]:
                    # Try to find each field and extract it
                    field_pattern = f'"{field}":\s*'
                    if field in ["facts", "actionable_items"]:
                        # Arrays
                        array_pattern = f'"{field}":\s*\[(.*?)\]'
                        match = re.search(array_pattern, repaired, re.DOTALL)
                        if match:
                            try:
                                data[field] = json.loads(f"[{match.group(1)}]")
                            except:
                                # If array parsing fails, try to extract items manually
                                items = re.findall(r'"([^"]+)"', match.group(1))
                                data[field] = items
                    else:
                        # Strings or objects
                        # Try to find the value - it could be a string or object
                        # For detailed_outline, if it's malformed, convert to string
                        if field == "detailed_outline":
                            # Try to extract the entire object as a string representation
                            obj_start = repaired.find(f'"{field}":')
                            if obj_start != -1:
                                obj_start = repaired.find('{', obj_start)
                                if obj_start != -1:
                                    # Find matching closing brace
                                    brace_count = 0
                                    obj_end = obj_start
                                    for i in range(obj_start, len(repaired)):
                                        if repaired[i] == '{':
                                            brace_count += 1
                                        elif repaired[i] == '}':
                                            brace_count -= 1
                                            if brace_count == 0:
                                                obj_end = i + 1
                                                break
                                    if obj_end > obj_start:
                                        # Extract and try to convert to string representation
                                        obj_str = repaired[obj_start:obj_end]
                                        try:
                                            data[field] = json.loads(obj_str)
                                        except:
                                            # If parsing fails, convert the structure to a readable string
                                            data[field] = str(obj_str).replace('{', '').replace('}', '').strip()
                        else:
                            # Regular string fields
                            string_pattern = f'"{field}":\s*"([^"]*)"'
                            match = re.search(string_pattern, repaired)
                            if match:
                                data[field] = match.group(1)
                
                # If we got at least some data, return it (with defaults for missing fields)
                if data:
                    # Set defaults for missing fields
                    for field in ["description", "summary", "minutes_of_meeting", "detailed_outline", "facts", "actionable_items"]:
                        if field not in data:
                            if field in ["facts", "actionable_items"]:
                                data[field] = []
                            else:
                                data[field] = ""
                    return data
            except Exception:
                pass
            pass
    
    # Final fallback: show the actual response for debugging
    print(f"\nDEBUG: Ollama response (first 500 chars):\n{content[:500]}", file=sys.stderr)
    if len(content) > 500:
        print(f"... (truncated, total length: {len(content)} chars)", file=sys.stderr)
    
    # Try to save the full response to a temp file for inspection
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
        tmp.write(content)
        tmp_path = tmp.name
        print(f"\nDEBUG: Full response saved to: {tmp_path}", file=sys.stderr)
    
    # Try one last aggressive repair attempt
    try:
        # Read the saved file and try to repair it
        with open(tmp_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        # More aggressive repairs
        repaired = file_content
        
        # Clean invalid control characters first
        repaired = clean_control_characters(repaired)
        repaired = balance_json_structure(repaired)
        
        # Remove trailing commas more aggressively
        repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
        
        # Try to fix unclosed strings (add closing quote if missing before comma/brace)
        # This is a last resort and might not always work
        
        # Try parsing with json.JSONDecoder which is more lenient
        decoder = json.JSONDecoder()
        try:
            result, idx = decoder.raw_decode(repaired)
            if idx > 0:  # If we parsed something
                print(f"\nDEBUG: Successfully parsed JSON after aggressive repair!", file=sys.stderr)
                return result
        except json.JSONDecodeError as e:
            # Show the exact error
            print(f"\nDEBUG: JSONDecoder error: {e}", file=sys.stderr)
            print(f"Error position: {e.pos}", file=sys.stderr)
            if e.pos < len(repaired):
                start = max(0, e.pos - 100)
                end = min(len(repaired), e.pos + 100)
                print(f"Context: {repaired[start:end]}", file=sys.stderr)
    except Exception as e:
        print(f"\nDEBUG: Final repair attempt failed: {e}", file=sys.stderr)
    
    # Show the actual content from the temp file
    try:
        with open(tmp_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        print(f"\nDEBUG: Full JSON content from temp file ({len(file_content)} chars):", file=sys.stderr)
        print(file_content, file=sys.stderr)
    except Exception as e:
        print(f"\nDEBUG: Could not read temp file: {e}", file=sys.stderr)
    
    raise RuntimeError(
        f"Failed to parse JSON from Ollama response after all repair attempts.\n"
        f"Response length: {len(content)} characters\n"
        f"First 200 chars: {content[:200]}\n"
        f"Last 200 chars: {content[-200:]}\n"
        f"Full response saved to: {tmp_path}\n"
        f"Please check the file above to see the exact JSON structure.\n"
        f"You can manually inspect and fix the JSON, then load it."
    )


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
        "You are an expert meeting summarizer. You MUST respond with valid JSON only, no other text. "
        "Read the conversation thoroughly and produce a comprehensive, detailed, factual, and actionable summary. "
        "Extract all relevant information, context, nuances, and details from the transcript. "
        "Do not add information not present in the transcript, but be thorough in capturing everything that was discussed. "
        "CRITICAL: Your response must be valid JSON that can be parsed directly. Do not include any explanatory text, markdown formatting, or code blocks."
    )
    user_prompt = (
        "Analyze the following conversation and respond with ONLY valid JSON (no markdown, no code blocks, no explanations). "
        "The JSON must match this exact schema:\n\n"
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
        "- detailed_outline: Create a comprehensive, in-depth hierarchical outline of EVERYTHING discussed. Include all topics, subtopics, sub-subtopics, discussions, decisions, key points, arguments, examples, data points, and details. Use clear hierarchical structure with sections, subsections, and bullet points. This should be extremely detailed and serve as a complete reference of the meeting content. CRITICAL: This must be a STRING value, not a nested object. Format the outline as plain text with markdown-style formatting (use # for headers, - for bullets, etc.) but return it as a JSON string value.\n\n"
        "- facts: Extract ALL objective facts, statements, data points, numbers, dates, names, locations, and verifiable information mentioned in the conversation. Be comprehensive - include every factual statement, statistic, reference, and concrete piece of information. This should be an extensive list.\n\n"
        "- actionable_items: Extract ALL tasks, action items, commitments, deadlines, responsibilities, and follow-ups mentioned. Include who is responsible (if mentioned), deadlines, specific deliverables, and any conditions or dependencies. Be thorough and capture every actionable item discussed.\n\n"
        "Remember: Be comprehensive, detailed, and thorough in EVERY section. Capture as much information as possible from the transcript.\n\n"
        f"Conversation:\n{normalized_text}\n\n"
        "IMPORTANT: Respond with ONLY the JSON object, starting with {{ and ending with }}. No markdown, no code fences, no explanations, no other text."
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



