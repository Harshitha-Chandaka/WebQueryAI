# import subprocess
# subprocess.run(["pip", "install", "beautifulsoup4==4.12.2"])

import os
import time
import streamlit as st
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(model="llama3-70b-8192", temperature=0.2, streaming = True)

st.title("WebQueryAI")
st.subheader("A smart bot for querying websites")
st.info("Enter the URL of a website and ask queries related to its content!")

# Initialize session state variables
if "scraped_websites" not in st.session_state:
    st.session_state.scraped_websites = {}

# Input: Website URL
url = st.text_input("Enter the website URL:")

if url:
    # Check if the website is already scraped
    if url in st.session_state.scraped_websites:
        documents = st.session_state.scraped_websites[url]["documents"]
        scrape_time = st.session_state.scraped_websites[url]["scrape_time"]
        vector_store = st.session_state.scraped_websites[url]["vector_store"]
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        st.success(f"Website already scraped! Scraped time: {scrape_time:.2f} seconds")
    else:
        st.info("Scraping website content...")
        scrape_start_time = time.time()

        try:
            #scraping content from the URL
            response = requests.get(url, timeout=10)  
            if response.status_code != 200:
                st.error(f"Error: Received status code {response.status_code}")
                st.stop()
            else:
                soup = BeautifulSoup(response.text, "html.parser")
                content = " ".join([tag.get_text() for tag in soup.find_all(["p", "h1", "h2", "h3", "li", "ol", "ul", "td"])]).strip()

                tables = soup.find_all("table")
                table_data = []
        
                for table in tables:
                    rows = table.find_all("tr")
                    table_text = []
                    for row in rows:
                        cols = row.find_all(["td", "th"])  # Extract both headers and data
                        # row_text = [col.get_text(strip=True) for col in cols]
                        row_text = ",".join([col.get_text(strip=True) for col in cols])
                        table_text.append("| " + " | ".join(row_text) + " |")  # Convert to Markdown Table format

                    if table_text:
                        table_data.append("\n".join(table_text))

            # Merge tables into content
                if table_data:
                    content += "\n\n### Extracted Tables:\n\n" + "\n\n".join(table_data)
                
                if not content:
                    st.error("Error: No extractable content found on the webpage.")
                    st.stop()

                scrape_time = time.time() - scrape_start_time
                st.success(f"Website content extracted successfully! Scrapped time: {scrape_time:.2f} seconds")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error scraping website: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

        document = Document(page_content=content, metadata={"source": url})

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=100)
        documents = text_splitter.split_documents([document])
        st.info(f"Number of chunks created: {len(documents)}")

        #FAISS with HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        st.session_state.scraped_websites[url] = {
                    "documents": documents,
                    "vector_store": vector_store,
                    "scrape_time": scrape_time
                }
        
        st.success("FAISS vector store created.")

user_query = st.text_input("Ask a question based on the content:")
if user_query:
    #retrieve relevant documents for the user's query
    relevant_docs = retriever.invoke(user_query)

    if relevant_docs:
        st.info(f"Retrieved {len(relevant_docs)} relevant document(s).")

        combined_content = "\n\n".join([doc.page_content for doc in relevant_docs])

        #limiting the combined content length to the model's token capacity
        max_combined_length = 3000
        combined_content = combined_content[:max_combined_length]

        #creating a ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
                        ("system", "Based on the content retrieved from the provided URL, answer the following question:"
                           "If the information is not explicitly mentioned, respond with:"
                           " 'Sorry! The related content is not available in the provided URL.'"),
                        ("system", "{content}"),
                        ("user", "Question: {question}")
        ])
                
        prompt_input = {
                "content": combined_content,
                "question": user_query
        }

        try:
            #generate the response 
            result = llm.invoke(prompt.format_prompt(**prompt_input).to_string()).content
            st.write("*Answer:*")
            st.write(result)
        except Exception as e:
            st.error(f"Error generating response: {e}")
    else:
        st.warning("No relevant documents found. Please refine your query.")