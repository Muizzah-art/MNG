import os
import re
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path

import soundfile as sf
import torch
import whisper
from pyannote.audio import Pipeline
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT

from dotenv import load_dotenv 

load_dotenv()


class MeetingNotesGenerator:
    """
    Loads Whisper + pyannote once, reuses them across calls.
    Instantiate this once per session/process instead of calling
    module-level functions repeatedly.
    """

    def __init__(self, hf_token: str, whisper_model_size: str = "base"):
        self.hf_token = hf_token
        self._whisper_model = None
        self._whisper_model_size = whisper_model_size
        self._diarization_pipeline = None
        self._openai_client = None

    @property
    def whisper_model(self):
        if self._whisper_model is None:
            t0 = time.time()
            self._whisper_model = whisper.load_model(self._whisper_model_size)
            print(f"[timing] Whisper model load: {time.time() - t0:.1f}s")
        return self._whisper_model

    @property
    def diarization_pipeline(self):
        if self._diarization_pipeline is None:
            t0 = time.time()
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.0",
                token=self.hf_token
            )
            print(f"[timing] Diarization model load: {time.time() - t0:.1f}s")
            if pipeline is None:
                raise RuntimeError(
                    "Failed to load pyannote/speaker-diarization-3.0. "
                    "Check that your HF token has accepted the model's "
                    "terms of use for both 'speaker-diarization-3.0' and "
                    "'segmentation-3.0' on huggingface.co."
                )
            self._diarization_pipeline = pipeline
        return self._diarization_pipeline

    @property
    def openai_client(self):
        if self._openai_client is None:
            if not os.environ.get("GROQ_API_KEY"):
                raise RuntimeError("GROQ_API_KEY environment variable is not set.")
            self._openai_client = OpenAI(
                api_key=os.environ.get("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            )
        return self._openai_client

    @staticmethod
    def _convert_to_wav(audio_path: Path) -> Path:
        """
        Converts any input audio (e.g. .m4a) to 16kHz mono .wav using ffmpeg.
        Sidesteps torchcodec's decode path entirely and also speeds up
        both Whisper and pyannote, since they don't have to decode m4a
        themselves.
        """
        tmp_wav = Path(tempfile.gettempdir()) / f"{audio_path.stem}_converted.wav"

        command = [
            "ffmpeg",
            "-y",              
            "-i", str(audio_path),
            "-ar", "16000",    
            "-ac", "1",        
            "-af", "loudnorm", 
            str(tmp_wav)
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to convert {audio_path} to wav:\n{result.stderr}"
            )

        return tmp_wav

    def get_chronological_dialogue(
        self,
        audio_path: str,
        num_speakers: int = None,
        language: str = "en",
        initial_prompt: str = None
    ) -> list:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        wav_path = self._convert_to_wav(audio_path)

        chronological_script = []

        t0 = time.time()
        whisper_result = self.whisper_model.transcribe(
            str(wav_path),
            language=language,
            initial_prompt=initial_prompt,
            beam_size=5,
            best_of=5
        )
        segments = whisper_result["segments"]
        print(f"[timing] Whisper transcription: {time.time() - t0:.1f}s")

        t0 = time.time()
        waveform, sample_rate = sf.read(str(wav_path), dtype="float32")
        if waveform.ndim == 1:
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)  
        else:
            waveform_tensor = torch.from_numpy(waveform.T) 

        diarization_result = self.diarization_pipeline(
            {"waveform": waveform_tensor, "sample_rate": sample_rate},
            num_speakers=num_speakers
        )
        print(f"[timing] Speaker diarization: {time.time() - t0:.1f}s")

        if hasattr(diarization_result, "itertracks"):
            diarization_annotation = diarization_result
        else:
            diarization_annotation = diarization_result.speaker_diarization

        print(f"[debug] Speakers found by diarization: {sorted(diarization_annotation.labels())}")

        for segment in segments:
            w_start = segment["start"]
            w_end = segment["end"]
            w_text = segment["text"].strip()

            if not w_text:
                continue

            best_speaker = "UNKNOWN"
            max_overlap = 0

            for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
                overlap_start = max(w_start, turn.start)
                overlap_end = min(w_end, turn.end)
                overlap_duration = overlap_end - overlap_start

                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = speaker

            chronological_script.append({
                "speaker": best_speaker,
                "text": w_text,
                "start": round(w_start, 1),
                "end": round(w_end, 1)
            })

        return chronological_script

    def summarize_meeting(self, meeting_text: list) -> str:
        if not meeting_text:
            raise ValueError("meeting_text is empty — nothing to summarize.")

        transcript_str = "\n".join(
            f"[{seg['start']}s - {seg['end']}s] {seg['speaker']}: {seg['text']}"
            for seg in meeting_text
        )

        prompt = textwrap.dedent("""
            You are an expert executive assistant. Your task is to analyze the provided chronological meeting transcript (which includes speaker IDs and timestamps) and generate a concise, high-utility meeting summary.

            Please structure your response into the following four clear sections:

            1. MEETING OBJECTIVE
            - A 1-2 sentence summary of the primary purpose of the meeting.

            2. KEY DISCUSSION POINTS
            - Break down the main topics discussed. Group related thoughts together, regardless of when they occurred in the transcript.
            - Format each topic as: **Topic Name**: Detailed bullet points summarizing what was said, debated, or agreed upon. Mention specific speakers by their ID (e.g., SPEAKER_00) when they make critical points.

            3. ACTION ITEMS
            - Create a bulleted list of specific tasks assigned during the meeting.
            - Use the format: `[Assignee (Speaker ID)]` Action item description. (Include any mentioned deadlines).
            - If an action item is clear but the assignee is ambiguous, label it `[Unassigned]`.

            4. DECISIONS MADE
            - A bulleted list of final conclusions, policy changes, or agreements reached during the call.

            CRITICAL INSTRUCTIONS:
            - Ignore filler words, casual greetings, small talk, and repetitive sentences.
            - Maintain strict factual accuracy based ONLY on the provided text. Do not hallucinate or assume context.
            - Keep speaker IDs intact so the team knows who said what.
        """).strip()

        t0 = time.time()
        response = self.openai_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Transcript:\n\n{transcript_str}"}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        print(f"[timing] Groq summarization: {time.time() - t0:.1f}s")

        return response.choices[0].message.content

    def save_summary_pdf(self, summary_text: str, output_path: str) -> str:
        """
        Converts the LLM's summary (plain text with **bold** and numbered/
        bulleted lines) into a formatted PDF using reportlab.
        """
        styles = getSampleStyleSheet()
        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            spaceBefore=14,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=10.5,
            leading=15,
            alignment=TA_LEFT,
        )
        bullet_style = ParagraphStyle(
            "Bullet",
            parent=body_style,
            leftIndent=16,
            bulletIndent=4,
        )

        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = [Paragraph("Meeting Summary", styles["Title"]), Spacer(1, 12)]

        section_pattern = re.compile(r"^\d+\.\s+[A-Z\s]+$")

        for raw_line in summary_text.splitlines():
            line = raw_line.strip()
            if not line:
                story.append(Spacer(1, 6))
                continue

            line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)

            if section_pattern.match(re.sub(r"<[^>]+>", "", line)):
                story.append(Paragraph(line, heading_style))
            elif line.startswith(("- ", "* ")):
                story.append(Paragraph(f"• {line[2:]}", bullet_style))
            else:
                story.append(Paragraph(line, body_style))

        doc.build(story)
        return output_path

    def save_transcript_pdf(self, meeting_text: list, output_path: str) -> str:
        """
        Saves the exact chronological transcript (speaker, timestamps, text)
        as a PDF — this is what was actually said, unedited by the LLM.
        """
        styles = getSampleStyleSheet()
        body_style = ParagraphStyle(
            "TranscriptLine",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            spaceAfter=6,
        )

        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = [Paragraph("Full Transcript", styles["Title"]), Spacer(1, 12)]

        for seg in meeting_text:
            line = f"<b>[{seg['start']}s - {seg['end']}s] {seg['speaker']}:</b> {seg['text']}"
            story.append(Paragraph(line, body_style))

        doc.build(story)
        return output_path

    def run(
        self,
        audio_path: str,
        num_speakers: int = None,
        language: str = "en",
        initial_prompt: str = None,
        transcript_output_path: str = "meeting_transcript.pdf",
        summary_output_path: str = "meeting_summary.pdf"
    ) -> dict:
        """Convenience method: transcribe + diarize + summarize + save both PDFs."""
        transcript = self.get_chronological_dialogue(
            audio_path,
            num_speakers=num_speakers,
            language=language,
            initial_prompt=initial_prompt
        )
        summary = self.summarize_meeting(transcript)
        transcript_pdf_path = self.save_transcript_pdf(transcript, transcript_output_path)
        summary_pdf_path = self.save_summary_pdf(summary, summary_output_path)
        return {
            "transcript": transcript,
            "summary": summary,
            "transcript_pdf_path": transcript_pdf_path,
            "summary_pdf_path": summary_pdf_path
        }


if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN environment variable before running.")

    generator = MeetingNotesGenerator(hf_token=hf_token, whisper_model_size="small")
    result = generator.run(
        "MNG.m4a",
        num_speakers=5
    )

    print(result["summary"])
    print(f"Saved transcript to: {result['transcript_pdf_path']}")
    print(f"Saved summary to: {result['summary_pdf_path']}")