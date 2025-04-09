import asyncio
from statemachine import StateMachine, State
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

# Define the states of the learning workflow
class LearningWorkflow(StateMachine):
    # Define the states
    start = State("Start", initial=True)
    study = State("Study")
    quiz = State("Quiz")
    finish = State("Finish", final=True)
    
    # Define transitions
    start_to_study = start.to(study)
    study_to_quiz = study.to(quiz)
    quiz_to_finish = quiz.to(finish)

    def __init__(self):
        super().__init__()
        self.executor = ThreadPoolExecutor()

    # Decorator to run state-entering methods in the executor
    def run_in_executor(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, func, *args, **kwargs)
            return result
        return wrapper

    # Apply the decorator to the on_enter_* methods
    @run_in_executor
    async def on_enter_study(self):
        print("Starting to study...")
        await asyncio.sleep(2)  # Simulate study time
        print("Study completed.")
    
    @run_in_executor
    async def on_enter_quiz(self):
        print("Taking quiz...")
        await asyncio.sleep(1)  # Simulate quiz time
        print("Quiz completed.")
    
    @run_in_executor
    async def on_enter_finish(self):
        print("Finishing lesson...")
        await asyncio.sleep(1)  # Simulate finish time
        print("Lesson finished.")
    

# A helper function to run the state machine in an executor
async def run_learning_workflow():
    workflow = LearningWorkflow()

    # Simulate running the learning workflow
    await workflow.start_to_study()
    await workflow.study_to_quiz()
    await workflow.quiz_to_finish()


# Running the workflow inside the executor
async def run_in_executor():
    task1 = asyncio.create_task(run_learning_workflow())
    task2 = asyncio.create_task(run_learning_workflow())

    await task1
    await task2


# Example usage
if __name__ == "__main__":
    asyncio.run(run_in_executor())
