#%%
import os
from dotenv import load_dotenv
import pdflow
from pdflow.document_analyzer import DocumentAnalyzer, TwoStageExtractor, YOLOSegmentationAdapter
from pdflow import extract_qrs_decoded
from openai import AzureOpenAI
load_dotenv()

# Test QR Code Extraction
test_pdf = './test1.pdf'
qr_codes = extract_qrs_decoded(test_pdf)
print("QR Code Sample:", qr_codes[0] if qr_codes else "No QR codes found")

#%%
# Document Intelligence config using Azure DocumentAnalysisClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

endpoint = os.getenv('ENDPOINT_DOCI')
key = os.getenv('KEY_DOCI')

document_analysis_client = DocumentAnalysisClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)

#%%
# Set up PydanticAI Agent using AsyncAzureOpenAI
from openai import AsyncAzureOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent

endpoint_gpt4o = os.getenv('ENDPOINT_GPT4O')
token_gpt4o = os.getenv('TOKEN_GPT4O')


client_gpt4o = AzureOpenAI(
    azure_endpoint=endpoint_gpt4o,
    api_key=token_gpt4o,
    api_version="2024-02-01",
)



model = OpenAIModel('gpt-4o', openai_client=client_gpt4o)
agent = Agent(model)

# Initialize two-stage extractor with pre-configured Agent
text_extraction_model = TwoStageExtractor(document_analysis_client, agent)

#%%
# Set up YOLO-based Segmentation Model
from ultralytics import YOLO

yolo_model = YOLO('../yolov11l_best.pt')
segmentation_model = YOLOSegmentationAdapter(yolo_model)

#%%
# Initialize DocumentAnalyzer with segmentation and extraction models
document_analyzer = DocumentAnalyzer(segmentation_model, text_extraction_model)

#%%
# Retrieve images by numeric label (class id)
images = document_analyzer.get_by_id(test_pdf, 9)
if images:
    print("Retrieved Image Sample:", images[0])
else:
    print("No images found for class id 9.")

# %%
segmentation_model.labels()
# %%
document_analyzer.get_by_label(test_pdf, 'Picture')[0]
# %%
extracted = document_analyzer.extract_fields(test_pdf, ['Actividad Economica'])
extracted
# %%
