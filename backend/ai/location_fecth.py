from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",
    task="text-generation",
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)


class fetch_location(BaseModel):
    Name: str = Field(description="Name of the dermatology hospital or clinic")
    Area: str = Field(description="Area or locality of the hospital")
    Reason: str = Field(description="One-line reason for recommendation")

parser = PydanticOutputParser(pydantic_object=fetch_location)
    


prompt = PromptTemplate(
    template='''
You are a medical recommendation assistant.

User Location: {location}
Health Concern: {topic}

Task:
Recommend ONLY ONE nearby dermatology hospital or clinic based on the given location.

Selection Criteria:
- Specializes in dermatology or skin care
- Good public reputation and reviews
- Qualified and experienced dermatologists
- Ethical and professional medical practice

Instructions:
- Recommend only ONE hospital or clinic (do not list multiple).
- If exact hospital data is unavailable, suggest a well-known and commonly trusted dermatology hospital for the region.
- Do NOT claim real-time data access or live review scraping.
- Keep the recommendation concise and professional.

Output format (strict):
Recommended Dermatology Hospital:
Name:
Area: 
Reason (1 short line explaining why it is recommended):

\n

{format_instruction}

''',
input_variables=["location","topic"],
partial_variables = {"format_instruction": parser.get_format_instructions()}  
)

chain = prompt | model | parser
def get_location(location,topic):
    return chain.invoke(
        {
            "location":location,
            "topic":topic
        })


