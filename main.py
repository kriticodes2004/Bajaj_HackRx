import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Load environment variables from .env
load_dotenv()

# Get the API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY is not set in the .env file.")

# Set up the Groq LLM with the correct model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"  # ✅ Use supported model
)

# Output parser
parser = JsonOutputParser()

# Prompt template
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("""
You are a structured query parser for natural language questions across domains like insurance, healthcare, legal, property, tax, pension, employment, and rights.

Your tasks:

1. Detect the main **domain** (e.g., "insurance", "legal", "tax", "employment", etc.).
2. If the domain is **insurance**, detect the **sub_domain**:
   - "health", "life", "car", "travel", "child", "education", "property", "accident", "fire", "home", "marine"

3. Extract **common_fields** (if available):
   - age (integer)
   - gender (e.g., "male", "female", "other")
   - location (city or region)
   - policy_duration (as a string, e.g., "3 months", "1 year")

4. Extract **domain_specific_fields** based on the domain:
   - For insurance/health: hospital_name, procedure, diagnosis, treatment_date, claim_type
   - For insurance/life: nominee_name, cause_of_death, sum_assured, policy_term
   - For insurance/car: vehicle_type, registration_number, accident_date, damage_type, garage_name
   - For legal: case_type, law_reference, jurisdiction, court_name
   - For tax: assessment_year, tax_category, deduction_type
   - For pension: retirement_age, pension_plan, contribution_period

🛑 Important:
- You MUST always include the top-level keys: "domain", "sub_domain", "common_fields", and "domain_specific_fields".
- If a field is unknown or missing, use an empty string or leave the sub-dictionary empty.
- Return **only valid JSON** without any explanation or formatting.

Return the result in this format:
{{
  "domain": "...",
  "sub_domain": "...",
  "common_fields": {{
    "age": ...,
    "gender": "...",
    "location": "...",
    "policy_duration": "..."
  }},
  "domain_specific_fields": {{
    ...
  }}
}}

Query: {query}
""")


# Build chain
chain = (
    {"query": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# Example query
query = "My car was in an accident last week in Pune. I’ve had a policy with HDFC Ergo for 2 years. It’s an SUV and the garage is Maruti Center. Claim hasn’t been processed."

# Run and print result
try:
    output = chain.invoke(query)
    print("Parsed Output:\n")
    print(json.dumps(output, indent=2))
except Exception as e:
    print("❌ Error during parsing:\n", e)
