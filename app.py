from config import VECTOR_DB_DIR
from rag_chain import build_rag_chain


def main():

    print("RAG Document Assistant Started")
    print("Type 'exit' to quit\n")

    qa_chain = build_rag_chain(VECTOR_DB_DIR)

    while True:

        question = input("User: ")

        if question.lower() == "exit":
            break

        result = qa_chain.invoke({
            "question": question
        })

        answer = result["answer"]

        print("\nAssistant:", answer)
        print()


if __name__ == "__main__":
    main()


# import ollama

# response = ollama.chat(
#     model="llama3",
#     messages=[
#         {"role": "user", "content": "Explain black holes simply"}
#     ],
# )

# print(response["message"]["content"])