#%%
import os
from dotenv import load_dotenv
import pdflow
from pdflow.document_analyzer import DocumentAnalyzer, TwoStageExtractor, YOLOSegmentationAdapter
from pdflow import extract_qrs_decoded
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
from openai import AsyncAzureOpenAI
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

test_pdf = 'tests/test1.pdf'
qr_codes = extract_qrs_decoded(test_pdf)
print("QR Code Sample:", qr_codes[0] if qr_codes else "No QR codes found")

endpoint = os.getenv('ENDPOINT_DOCI')
key = os.getenv('KEY_DOCI')

document_analysis_client = DocumentAnalysisClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("TOKEN_GPT4O")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("ENDPOINT_GPT4O")

print(f"Using Azure OpenAI endpoint: {os.environ['AZURE_OPENAI_ENDPOINT']}")

client = AsyncAzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01"
)

model = OpenAIModel('gpt-4o-pdf', openai_client=client)
agent = Agent(model)

text_extraction_model = TwoStageExtractor(document_analysis_client, agent)

from ultralytics import YOLO

yolo_model = YOLO('yolov11l_best.pt')
segmentation_model = YOLOSegmentationAdapter(yolo_model)

document_analyzer = DocumentAnalyzer(segmentation_model, text_extraction_model)

async def main():
    extracted = document_analyzer.extract_fields(test_pdf, ['RFC:', 'Primer Apellido', 'Actividad Economica'])
    print(await extracted)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
