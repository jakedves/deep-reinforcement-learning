# Set seeds before doing anything else
USE_SEED = True
SEED = 235

"""
Runtime Environment
"""
COLAB = False
PROFILE_CODE = False
PROFILE_FILENAME = "results.profile"


"""
RainbowDQN Options
"""
# DoubleDQN is our default agent
PRIORITISED_EXPERIENCE_REPLAY = True
N_STEP = 5  # Set to > 1 to use n-step TD(n) update, otherwise standard TD(0) update is used
DUELING = True
NOISY_NETWORKS = True  # [FACTORISED, NONE] (initially planned to also include Independent)
DISTRIBUTIONAL = False
# Rainbow is when all the above are on

# Random Network Distillation
RND = False
INITIALISATION_STEPS = 200
INTRINSIC_DISC_RATE = 0.999

# Parameters for Prioritised Experience Replay (recommended in the paper)
PER_ALPHA = 0.6
PER_BETA = 0.4
PER_EPSILON = 0.1

# Parameters for NoisyNetworks
INITIAL_NOISE_STRENGTH = 2.5

# Distributional DQN parameters,
V_MIN = -10.0
V_MAX = 10.0
NUM_ATOMS = 51  # from the paper

"""
RL-Agent Hyperparameters
"""
# Type of Function Approximator to use
NETWORK_TYPE = NetworkTypes.CNN  # DNN, CNN
ADAM_EPSILON = 1.5e-4

# Settings mainly affecting convergence and time taken
# Steps are far more fair than episodes, means each agent will get a fair amount of training time
NUMBER_OF_STEPS = 5_000_000
LR = 0.00025
DISCOUNT_FACTOR = 0.9
COPY_NETWORK_FREQUENCY = 10_000  # How frequently we copy the online -> target network
EXPERIENCE_REPLAY_BUFFER_SIZE = 10_000
EXPERIENCE_SAMPLE_SIZE = 32

# Values for exploration decay (when not using NoisyNets)
EPSILON = 1.0
EPSILON_DECAY = 0.9999
EPSILON_MIN = 0.01

# What episode to swap to the '3 Lives, All Levels' environment (0 = OFF)
# Ideally use a MULTIPLE of RECORD_FREQUENCY: This will mean you record your first transfer learning episode
TRANSFER_LEARNING_STEP_COUNT = 0
