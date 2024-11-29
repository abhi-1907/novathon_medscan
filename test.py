import os
import pytesseract
from PIL import Image
import re
from llmware.models import ModelCatalog

class PatientRecordProcessor:
    def __init__(self, gemini_api_key):
        """Initialize with specific Tesseract path and Gemini API key"""
        pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'
        self.model_catalog = ModelCatalog()
        self.gemini_api_key = gemini_api_key  # Gemini API key

    def ocr_image(self, image_path):
        """Extract text from image using Tesseract OCR"""
        try:
            image = Image.open(image_path)
            extracted_text = pytesseract.image_to_string(image)
            return extracted_text
        except Exception as e:
            print(f"OCR Error for {image_path}: {e}")
            return ""

    def extract_medical_info(self, text):
        """Extract medical information using basic parsing"""
        return {
            "Raw Text": text,
            "Patient Name": self._extract_patient_name(text),
            "Date": self._extract_date(text),
            "Diseases": self._extract_diseases(text),
            "Medications": self._extract_medications(text)
        }

    def _extract_patient_name(self, text):
        """Basic patient name extraction"""
        name_patterns = [
            r'Patient Name[:\s]*([A-Z][a-z]+ [A-Z][a-z]+)',
            r'Name[:\s]*([A-Z][a-z]+ [A-Z][a-z]+)'
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return "Name not found"

    def _extract_date(self, text):
        """Extract date using regex"""
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{4}-\d{1,2}-\d{1,2}'
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return "Date not found"

    def _extract_diseases(self, text):
        """Extract potential disease names"""
        disease_keywords = [
            'diabetes', 'hypertension', 'cancer', 'flu', 
            'pneumonia', 'infection', 'syndrome', 'disorder'
        ]
        return [word for word in disease_keywords if word in text.lower()]

    def _extract_medications(self, text):
        """Extract potential medication names"""
        medication_keywords = [
            'aspirin', 'insulin', 'antibiotics', 'paracetamol', 
            'ibuprofen', 'metformin', 'vaccine'
        ]
        return [word for word in medication_keywords if word in text.lower()]

    def classify_with_slim(self, extracted_text):
        """Classify extracted text using SLiM model from ModelCatalog"""
        llm_toolkit = self.model_catalog.get_llm_toolkit()
        
        # Test or use a specific SLiM model for classification
        result = self.model_catalog.tool_test_run("slim-sentiment-tool")  # Modify this for your needs
        
        return result

    def process_patient_records(self, image_directory):
        """Process patient record images in a directory"""
        if not os.path.exists(image_directory):
            print(f"Directory not found: {image_directory}")
            return

        files = os.listdir(image_directory)
        if not files:
            print(f"No files found in directory: {image_directory}")
            return

        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                image_path = os.path.join(image_directory, filename)
                
                # OCR Text Extraction
                extracted_text = self.ocr_image(image_path)
                
                # Extract Medical Information
                medical_info = self.extract_medical_info(extracted_text)
                
                # Classify using SLiM
                classification_result = self.classify_with_slim(extracted_text)
                
                # Print Results
                print(f"\nProcessing Image: {filename}")
                for key, value in medical_info.items():
                    print(f"{key}: {value}")
                print(f"Classification Result: {classification_result}")
                print("-" * 50)

def main():
    gemini_api_key = "AIzaSyD2alKChT_ovlqRgzCYPUlN5kl_9dCBcwA"  # Replace with your Gemini API key
    processor = PatientRecordProcessor(gemini_api_key)
    processor.process_patient_records(r'D:\novathonnew\records')

if __name__ == "__main__":
    main()
