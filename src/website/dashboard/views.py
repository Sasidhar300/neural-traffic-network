from django.contrib.auth import logout, login, authenticate
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from src.config.path_config import PRETRAINED_MODELS_DIR
from src.config.data_config import map_configs, signal_configs
from django.conf import settings
from src.main import main
from src.config.simulation_config import SIM_CONFIG, SimulationConfig, HyperParameters, RewardParameters # Import all needed config classes
# from .models import MapConfig, SignalConfig
import sys
import threading
import queue
import traceback
import time
from django.http import StreamingHttpResponse
from .simulation_tracker import simulation_tracker
import json
from pathlib import Path


# Create global output queue for streaming
output_queue = queue.Queue()
def home_view(request):
    return render(request, "home.html")


@login_required
def simulation_view(request):
    context = SIM_CONFIG.copy()
    context['maps'] = list(map_configs)
    context['is_simulation_running'] = simulation_tracker.is_running()
    context['is_simulation_complete'] = simulation_tracker.is_complete()
    context['is_simulation_error'] = simulation_tracker.is_error()
    return render(request, 'simulation.html', context=context)


@login_required
def run_simulation_view(request):
    if simulation_tracker.is_running():
        messages.error(request, "A simulation is already in progress. Please wait until it finishes.")
        return redirect("simulation")

    if request.method == "POST":
        try:
            # Reset global state for new run
            if not simulation_tracker.start_simulation():
                messages.error(request, "Failed to start: A simulation appears to be running.")
                return redirect("simulation")
            while not output_queue.empty():
                output_queue.get()

            # Collect preferences from the form
            preferences = {
                "display_gui": request.POST.get("display_gui") == "true",
                "load_model": request.POST.get("option") == "load_model",
                "load_pretrained_model": False,
                "training_episodes": int(request.POST.get("training_episodes", 0)),
                "map_name": request.POST.get("map_name"),
                "use_libsumo": False,
                "max_observation_distance": int(request.POST.get("max_observation_distance", 0)),
                "random_seed": int(request.POST.get("random_seed", 0)),

                "hyperparameters": {
                    "learning_rate": float(request.POST.get("learning_rate", 0.0)),
                    "optimizer_epsilon": float(request.POST.get("optimizer_epsilon", 0.0)),
                    "policy_clip_range": float(request.POST.get("policy_clip_range", 0.0)),
                    "value_clip_range": float(request.POST.get("value_clip_range", 0.0)),
                    "update_interval": int(request.POST.get("update_interval", 0)),
                    "minibatch_size": int(request.POST.get("minibatch_size", 0)),
                    "ppo_epochs": int(request.POST.get("ppo_epochs", 0)),
                    "standardize_advantages": request.POST.get("standardize_advantages")== "true",
                    "entropy_coefficient": float(request.POST.get("entropy_coefficient", 0.0)),
                    "max_gradient_norm": float(request.POST.get("max_gradient_norm", 0.0)),
                },

                "reward_params": {
                    "waiting_time_penalty": float(request.POST.get("waiting_time_penalty", 0.0)),
                    "throughput_reward": float(request.POST.get("throughput_reward", 0.0)),
                    "journey_reward": float(request.POST.get("journey_reward", 0.0)),
                    "queue_penalty": float(request.POST.get("queue_penalty", 0.0)),
                },
            }

            # Validate training episodes
            if preferences["training_episodes"] < 1:
                preferences["training_episodes"] = 3

            # Handle display_gui and use_libsumo - mutual exclusivity
            if preferences["display_gui"]:
                preferences["use_libsumo"] = False

            # Update the global SIM_CONFIG with all user preferences - only needed for future form defaults
            for key, value in preferences.items():
                if key in SIM_CONFIG and not isinstance(value, dict):
                    SIM_CONFIG[key] = value

            # Update nested dictionaries
            for key in ["hyperparameters", "reward_params"]:
                if key in preferences and key in SIM_CONFIG:
                    for subkey, value in preferences[key].items():
                        if subkey in SIM_CONFIG[key]:
                            SIM_CONFIG[key][subkey] = value

            # Get map and signal configs from database
            map_config = map_configs[preferences["map_name"]]
            signal_config = signal_configs[preferences["map_name"]]


            # Create specialized configuration objects
            hyperparameters = HyperParameters(**preferences["hyperparameters"])
            reward_params = RewardParameters(**preferences["reward_params"])

            # Create the full SimulationConfig
            simulation_config = SimulationConfig(
                pretrained_models_directory = PRETRAINED_MODELS_DIR / preferences["map_name"],
                display_gui=preferences["display_gui"],
                load_model=preferences["load_model"],
                load_pretrained_model=preferences["load_pretrained_model"],
                training_episodes=preferences["training_episodes"],
                map_name=preferences["map_name"],
                max_observation_distance=preferences["max_observation_distance"],
                random_seed=preferences["random_seed"],
                use_libsumo=preferences["use_libsumo"],
                hyperparameters=hyperparameters,
                reward_params=reward_params,
                map_config=map_config,
                signal_config=signal_config,
            )

            # Start simulation in a background thread
            thread = threading.Thread(target=run_simulation_in_thread, args=(simulation_config,))
            thread.daemon = True
            thread.start()

            # Redirect to output page that will show streaming output
            return redirect("simulation_output")

        except Exception as e:
            error_message = f"Error during simulation setup: {str(e)}\n{traceback.format_exc()}\n"
            output_queue.put(error_message)
            simulation_tracker.append_error(error_message)
            simulation_tracker.stop_simulation()
            cached_output = simulation_tracker.get_current_logs()
            return render(
                request,
                "simulation_output.html",
                {
                    "cached_output": cached_output, # Show logs leading up to error
                },
            )

    # If not a POST request, redirect back to simulation page
    return redirect("simulation")


def run_simulation_in_thread(simulation_config):
    """Run the simulation in a background thread and put output in queue"""
    # Custom stdout to capture output
    class QueuedOutput:
        def write(self, text):
            # Ensure each message ends with exactly one newline
            if text and not text.endswith("\n"): text += "\n"
            output_queue.put(text)
            simulation_tracker.append_log(text)
            sys.__stdout__.write(text)

        def flush(self):
            sys.__stdout__.flush()

    try:
        # Redirect stdout to our queued output
        sys.stdout = QueuedOutput()

        # Run the simulation
        main(simulation_config)

    except Exception as e:
        msg = f"Error in simulation: {str(e)}\n{traceback.format_exc()}"
        output_queue.put(msg)
        simulation_tracker.append_error(msg)
        simulation_tracker.stop_simulation()
    finally:
        # Reset stdout and mark simulation as complete

        simulation_tracker.stop_simulation()
        simulation_tracker.reset()
        sys.stdout = sys.__stdout__

@login_required
def simulation_output_view(request):
    """View for the page that displays streaming output"""
    if not simulation_tracker.is_running() and not simulation_tracker.is_complete():
        return redirect("simulation")

    context = {
        'cached_output': simulation_tracker.get_current_logs() if not simulation_tracker.just_started() else '',
        'is_simulation_running': simulation_tracker.is_running(),
    }
    return render(request, "simulation_output.html", context)

@login_required
def sse_updates(request):
    """Server-Sent Events endpoint for streaming simulation output"""

    def event_stream():
        while True:
            # Check if there's new output
            try:
                # Non-blocking get with timeout
                message = output_queue.get(block=True, timeout=0.1)
                # Format as SSE with appropriate escaping
                if message:
                    # Escape newlines for SSE transmission
                    formatted_message = message.replace('\n', '\\n')
                    yield f"data: {formatted_message}\n\n"
            except queue.Empty:
                # No message available
                if not simulation_tracker.is_running() and simulation_tracker.is_complete() and not simulation_tracker.is_error():
                    yield "data: [SIMULATION_COMPLETE]\n\n"
                    break
                elif not simulation_tracker.is_running() and simulation_tracker.is_complete() and simulation_tracker.is_error():
                    yield "data: [SIMULATION_ERROR]\n\n"
                    break
                # Send a keepalive comment to maintain connection
                yield ": keepalive\n\n"
                time.sleep(0.5)

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"  # Disable buffering in Nginx
    return response

def register_view(request):
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        username = request.POST["username"].strip()
        email = request.POST["email"].strip()
        password = request.POST["password"]
        confirm_password = request.POST["confirm_password"]

        if not (username and password and confirm_password):
            messages.error(request, "Required fields are missing.")
            return render(request, "register.html")

        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return render(request, "register.html")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return render(request, "register.html")

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email is already registered.")
            return render(request, "register.html")

        user = User.objects.create_user(username=username, email=email, password=password)
        user.save()
        messages.success(request, "Account created successfully.")
        return render(request, "register.html")
    return render(request, "register.html")

def about_view(request):
    return render(request, 'about.html')

def signin_view(request):
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        username = request.POST["username"].strip()
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("home")
        else:
            messages.error(request, "Invalid username or password.")
            return render(request, "signin.html")

    return render(request, "signin.html")

def signout_view(request):
    logout(request)
    return redirect("signin")

@login_required
def result_view(request):
    if simulation_tracker.is_running() and not simulation_tracker.is_complete():
        return redirect("simulation_output")

    if not simulation_tracker.is_running() and not simulation_tracker.is_complete():
        return redirect("simulation")

    if simulation_tracker.is_error():
        return redirect("simulation_output")

    plots_dir = Path(settings.MEDIA_ROOT) / 'plots'
    metrics_dir = Path(settings.MEDIA_ROOT) / 'metrics'

    graph_data = []
    metrics_data = {}

    # Plot comparision plots
    comparisons_dir = plots_dir / 'comparisons'
    column_name = {"rewards" : "Rewards",
                    "waiting_times" : "Waiting Time (s)",
                    "queue_lengths" : "Queue Length (vehicles)",
                    "co2_emissions" : "CO2 Emissions (g)",
                    "avg_speeds" : "Average Speed (km/h)",
                    "fuel_consumption" : "Fuel Consumption (L/min)",
                    "noise_emission" : "Noise Emission (dB)"
                }

    if comparisons_dir.exists() and comparisons_dir.is_dir():
        # Collect all .png files in the comparisons folder
        for file in comparisons_dir.glob('*.png'):
            file_name = file.stem  # Get the file name without extension
            parts = file_name.rsplit('_', 2)
            if len(parts) == 3:
                comparison_name = column_name.get(parts[0])
                if comparison_name is None:
                    continue
                load_type = parts[1]
                comparison_entry = next((item for item in graph_data if item["name"] == comparison_name), None)
                if not comparison_entry:
                    comparison_entry = {"name": comparison_name, "light": "", "normal": "", "heavy": ""}
                    graph_data.append(comparison_entry)
                if load_type.lower() == "light":
                    comparison_entry["light"] = f"plots/comparisons/{file.name}"
                elif load_type.lower() == "normal":
                    comparison_entry["normal"] = f"plots/comparisons/{file.name}"
                elif load_type.lower() == "heavy":
                    comparison_entry["heavy"] = f"plots/comparisons/{file.name}"

    if metrics_dir.exists() and metrics_dir.is_dir():
        # Path to the performance_metrics.json file
        metrics_file = metrics_dir / "performance_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)

    return render(request, 'result.html', {
        'graph_data': graph_data,
        'media_url': settings.MEDIA_URL,
        'metrics_data': metrics_data
    })
