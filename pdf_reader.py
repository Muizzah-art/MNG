from pypdf import PdfReader

reader = PdfReader("meeting_transcript.pdf")

for page in reader.pages:
    print(page.extract_text())