from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


#llm = OpenAI()
llm2 = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
        You are a research assistant that will help generate a research paper.
        Answer the user query and use neccessary tools. 
        Wrap the output in this format and provide no other text\n{format_instructions}
     """
     ),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"), 
    ("placeholder", "{agent_scratchpad}"),
]).partial(
    format_instructions=parser.get_format_instructions()
)

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm2,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can I help you research in?\n")
raw_response = agent_executor.invoke({"query": query})
print(raw_response)

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print(f"Error parsing response: {e}")
    print(raw_response.get("output")[0]["text"])



#response_chatGPT = llm.responses.create(model="gpt-4.1", input="Was Obito right to start the war for the girl who did not like him and kill the woman who loved him like a mother?")
#response_Claude = llm2.invoke("Was Obito right to start the war for the girl who did not like him and kill the woman who loved him like a mother?")

#print(response_chatGPT.output_text)
#print("--------------------------------")
#rint(response_Claude.content)


