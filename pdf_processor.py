# Install libraries if needed: pip install PyPDF2 pdfplumber pandas
import PyPDF2
import pandas as pd
import re

# Step 1: Extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:

        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Step 2: Basic text cleaning
def clean_text(text):
    # Remove extra whitespace, newlines, and common noise
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
    text = text.strip()  # Remove leading/trailing whitespace
    return text

# Step 3: Process and save the text
def process_pdf(pdf_path, output_csv):
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    
    # Store in a DataFrame and save to CSV
    df = pd.DataFrame({'text': [cleaned_text]})
    df.to_csv(output_csv, index=False)
    return cleaned_text

# Example usage
pdf_path = "annualreport-2023.pdf"  # Replace with your PDF file path
output_csv = "processed_text.csv"
cleaned_text = process_pdf(pdf_path, output_csv)

print("First 500 characters of cleaned text:")
print(cleaned_text[:500])