import os
import json
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
from openai import AsyncAzureOpenAI

load_dotenv()

async def test_azure_connection():
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

    # Test the extraction
    sample_text = """
    Name: John Doe
    Age: 30 years
    Occupation: Software Engineer
    """
    
    format_example = {
        "Name": "<Name>",
        "Age": "<Age>",
        "Email": "<Email>"
    }

    prompt = f"""You are a professional data extraction system.
    Extract the following fields from the text: Name, Age, Email
    Return ONLY valid JSON format with the extracted values.
    Use null for missing fields.
    Example format: {json.dumps(format_example, indent=2)}

    DOCUMENT CONTENT:
    {sample_text}"""

    result = await agent.run(prompt)
    print("\nExtracted fields:", result.data)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_azure_connection()) 