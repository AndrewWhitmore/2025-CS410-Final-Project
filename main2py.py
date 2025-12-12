import logging
from agent import create_agent
from utils.retriever import create_retriever
from utils.config import Config
from utils.state import GraphState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_passage(passage, index):
    if isinstance(passage, str):
        text = passage
        source = 'Unknown'
        score = 0
    else:
        source = passage.get('source', 'Unknown')
        score = passage.get('score', 0)
        text = passage.get('text', '')

    if len(text) > 200:
        text = text[:200] + "..."

    return f"Passage {index + 1}:\nText: \033[30;43m{text}\033[0m\n"




def main():
    try:
        # Load configuration
        config = Config()
        logger.info(f"Loaded configuration: {config}")

        # Initialize components
        retriever = create_retriever()

        # Create agent
        agent = create_agent()

        while True:
            # Prompt user for question
            question = input("Please enter your question (or type 'exit' to quit): ")

            if question.lower() == 'exit':
                print("Exiting the program. Goodbye!")
                break

            logger.info(f"Running agent with question: {question}")

            # Initialize the GraphState with all required fields
            initial_state = GraphState(
                question=question,
                context=[],
                current_step="",
                final_answer="",
                retriever=retriever,
                web_search_tool=None,
                error=None,
                selected_namespaces=[],
                web_search_results=[]
            )

            result = agent.invoke(initial_state)
            logger.info(f"Agent result: {result}")

            # ===== Task 2: this is where you can improve the user-facing output =====
            # Print retrieved passages
            if result.get("context"):
                print("\nRelevant passages found:")
                for i, passage in enumerate(result["context"]):

            # apply formatting
                    print(format_passage(passage, i))
                print("-" * 30)

            # Print final answer
            if result.get("final_answer"):
                print(f"\nAnswer: \033[1m{result['final_answer']}\033[0m")
            elif result.get("error"):
                print(f"\nError occurred: {result['error']}")
            else:
                print("\nNo answer or error was returned.")

            print("\n" + "-" * 50 + "\n")  # Add a separator between questions

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    main()
