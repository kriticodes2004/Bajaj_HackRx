import os
from dotenv import load_dotenv
from langchain_chroma import Chroma  # ✅ Updated import
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated import
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Load env variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Init Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama3-70b-8192"
)

# Load ChromaDB and Embeddings
persist_directory = "chromadb"
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

retriever = vectordb.as_retriever(search_kwargs={"k": 8})  # increased k for better match

# 1️⃣ User Input
query = input("Enter your insurance query: ").strip()

# 2️⃣ Query Parsing
parse_prompt = PromptTemplate.from_template("""
Parse the following insurance-related query and extract:
- domain (e.g., insurance, legal, healthcare)
- sub_domain (e.g., health, car, life, property, etc.)
- common_fields (age, gender, location, policy_duration)
- domain_specific_fields (procedure, hospital, damage_type, vehicle_type, etc.)

Respond in this JSON format:
{{
  "domain": "...",
  "sub_domain": "...",
  "common_fields": {{ }},
  "domain_specific_fields": {{ }}
}}

Query: {query}
""")

structured_query = llm.invoke(parse_prompt.format(query=query))
print("\n🧠 Parsed Query:")
print(structured_query.content)

# 3️⃣ Retrieve Relevant Clauses
print("\n🔍 Retrieving relevant clauses...\n")
retrieved_docs = retriever.get_relevant_documents(query)

print(f"📄 Retrieved {len(retrieved_docs)} clauses")

# Check if no relevant clauses
if not retrieved_docs:
    print("⚠️ No relevant clauses found. Please check if the PDFs are embedded correctly or try a simpler query.")
    exit()

# Show top 3 for debugging
for i, doc in enumerate(retrieved_docs[:3]):
    print(f"\n📄 [Doc {i+1}] Source: {doc.metadata.get('source', 'unknown')}")
    print(doc.page_content[:300] + "...\n")

# 4️⃣ Evaluate & Justify
eval_prompt = PromptTemplate.from_template("""
You are an insurance claims assistant.

Based on the user's structured query and the following policy clauses, determine:
- Is the requested procedure or event covered?
- If yes, what is the payout (assume ₹1,00,000 if not specified)?
- Justify the decision by referencing matching clause text.

Respond in JSON like:
{{
  "decision": "approved/rejected",
  "amount": "₹...",
  "justification": {{
    "matched_clauses": [
      {{
        "clause_text": "...",
        "source": "document name - page X"
      }}
    ],
    "reasoning": "..."
  }}
}}

Structured Query:
{structured_query}

Policy Clauses:
{clauses}
""")

clause_texts = "\n\n".join(
    [f"{i+1}. {doc.page_content.strip()} (source: {doc.metadata.get('source', 'unknown')})"
     for i, doc in enumerate(retrieved_docs)]
)

evaluation = llm.invoke(
    eval_prompt.format(
        structured_query=structured_query.content,
        clauses=clause_texts
    )
)

print("\n✅ Final Answer:")
print(evaluation.content)
