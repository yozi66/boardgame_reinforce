from generals.agents import RandomAgent, ExpanderAgent
from generals.envs import PettingZooGenerals
from my_expander_agent import MyExpanderAgent
from gradient_agent import GradientAgent

# Initialize agents
random = RandomAgent()
expander = ExpanderAgent()
myExpander = MyExpanderAgent()
gradient = GradientAgent()
gradient2 = GradientAgent(id="Gradient2")

# Names are used for the environment
agent_names = [random.id, expander.id]
agent_names = [expander.id, myExpander.id]
agent_names = [myExpander.id, gradient.id]
agent_names = [gradient.id, expander.id]
agent_names = [gradient.id, gradient2.id]

# Store agents in a dictionary
agents = {
    random.id: random,
    expander.id: expander,
    myExpander.id: myExpander,
    gradient.id: gradient,
    gradient2.id: gradient2,
}

# Create environment
env = PettingZooGenerals(agents=agent_names, render_mode="human")
observations, info = env.reset()

terminated = truncated = False
while not (terminated or truncated):
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = agents[agent].act(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    env.render()
input("Press Enter to exit...")
