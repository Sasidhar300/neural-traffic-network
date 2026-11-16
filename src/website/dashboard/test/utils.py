from secrets import token_urlsafe

TEST_USER = "ntn-test-user"
TEST_PASSWORD = token_urlsafe(18)

# request keys
USERNAME_KEY = "username"
EMAIL_KEY = "email"
PASSWORD_KEY = "password"
CONFIRM_PASSWORD_KEY = "confirm_password"


# new credentials
NEW_TEST_USER = "ntn-new-test-user"
NEW_TEST_PASSWORD = token_urlsafe(18)
NEW_TEST_EMAIL = "new-user@example.invalid"
NEW_INCORRECT_PASSWORD = token_urlsafe(18)

# INVALID credentials
INVALID_USERNAME_KEY = "unknown-user"
INVALID_PASSWORD_KEY = token_urlsafe(18)

# view names
HOME_TEMPLATE = "home.html"
REGISTER_TEMPLATE = "register.html"
RESET_TEMPLATE = "reset.html"
SIGNIN_TEMPLATE = "signin.html"
ABOUT_TEMPLATE = "about.html"
SIMULATION_TEMPLATE = "simulation.html"
SIMULATION_OUTPUT_TEMPLATE = "simulation_output.html"
RESULT_TEMPLATE = "result.html"

# URL names
HOME_URL = "/"
REGISTER_URL = "/register/"
SIGNIN_URL = "/signin/"
SIGNOUT_URL = "/signout/"
ABOUT_URL = "/about/"
RESULT_URL = "/result/"
SIMULATION_URL = "/simulation/"
RUN_SIMULATION_URL = "/run-simulation/"
SIMULATION_OUTPUT_URL = "/simulation-output/"
SSE_UPDATES_URL = "/sse-updates/"

# redirect
NEXT = "?next="

# special vars
EMPTY_STRING = ""
RUN_SIMULATION_DATA = {
                "map_name": "synthetic-cross",
                "option": "train_model",
                "training_episodes": "3",
                "max_observation_distance": "200",
                "random_seed": "7",
                "learning_rate": "0.0025",
                "optimizer_epsilon": "0.00001",
                "policy_clip_range": "0.1",
                "value_clip_range": "0.1",
                "update_interval": "256",
                "minibatch_size": "128",
                "ppo_epochs": "6",
                "standardize_advantages": "true",
                "entropy_coefficient": "0.001",
                "max_gradient_norm": "0.5",
                "waiting_time_penalty": "1",
                "throughput_reward": "1",
                "journey_reward": "1",
                "queue_penalty": "1",
                "plot_rewards": "true",
                "plot_waiting_times": "true",
                "plot_co2_emissions": "true",
                "plot_comparisons": "true",
            }
