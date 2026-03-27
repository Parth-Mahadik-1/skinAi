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

class report(BaseModel):
    introduction: str = Field( description="Definition and overview of the topic")
    causes: List[str] = Field( description="List of causes for the acne type")
    symptoms: List[str] = Field( description="Appearance and symptoms")
    prevention: List[str] = Field( description="Methods to prevent this condition")
    treatment: str = Field( description="Treatment plan based on user type")
    conclusion: str = Field( description="Final summary")

parser = PydanticOutputParser(pydantic_object=report)

prompt = PromptTemplate(
    template = """
Generate a concise, medically accurate, one-page clinical report on "{topic}".

User Type: {user_type}

Follow this structure strictly and keep content brief and relevant.

1. Introduction  
   - Define "{topic}" in 1 sentence.  
   - Brief formation mechanism (1 short sentence).  
   - Adjust language complexity based on user type.

2. Causes  
   - List key causes in 2–3 bullet points.  
   - Expert User: include limited scientific terms (e.g., P. acnes, inflammation).

3. Symptoms & Appearance  
   - Describe appearance in 2 short lines.  
   - Use dermatological terms only for Expert User.

4. Prevention  
   - 3 concise bullet points (skincare + lifestyle).

5. Treatment  
   - Normal User:
     - General OTC care only (no prescription drugs).
     - Add dermatologist consultation line.
   - Expert User:
     - Mention standard clinical treatments briefly.
     - Include 2–3 advanced modalities (laser, LED, chemical peels, AI tools).

6. Conclusion  
   - Normal User: short reassurance + referral (2 lines max).
   - Expert User: clinical summary in 1 line.

Constraints:
- Keep total length suitable for ONE A4 PDF page.
- Avoid unnecessary explanations.
- No unsafe or prescription advice for Normal Users.
- Maintain a professional medical tone.

Output only the final report text.

{format_instruction}


""",
input_variables=['topic','user_type'],
partial_variables = {"format_instruction": parser.get_format_instructions()}  
)

chain = prompt | model | parser

def generate_report(topic, user_type):
    return chain.invoke({
        "topic": topic,
        "user_type": user_type
    })
