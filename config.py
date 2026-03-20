import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
N_ROLLOUTS = 6    # rollouts per problem (paper uses group sampling)
N_PROBLEMS = 15   # problems to run
