import asyncio

from statemachine import State, StateMachine


class TestStateMachine(StateMachine):
    """Test State Machine."""

    state_1 = State("State 1", initial=True)
    state_2 = State("State 2")
    state_3 = State("State 3")
    state_4 = State("State 4")

    transition_1 = state_1.to(state_2) | state_2.to(state_1)
    transition_2 = state_2.to(state_3) | state_3.to(state_2)
    transition_3 = state_3.to(state_4) | state_4.to(state_3)

    def __init__(self, *args, **kwargs):
        self.state_changed = asyncio.Event()
        super().__init__(*args, **kwargs)

    async def on_enter_state_2(self):
        self.send("transition_2")

    async def on_enter_state_3(self):
        self.send("transition_3")

    def on_transition(self, event, state):
        print(f"Transitioning to {state} on event {event}")
        self.state_changed.set()
        self.state_changed.clear()

async def run_state_machine():
    """Run the state machine."""
    fsm = TestStateMachine()
    await fsm.activate_initial_state()
    print(f"Current state: {fsm.current_state}")
    await fsm.send("transition_1")
    print(f"Current state: {fsm.current_state}")


if __name__ == "__main__":
    # Example usage
    asyncio.run(run_state_machine())
