// JavaScript specific to the simulation page

document.addEventListener('DOMContentLoaded', function () {
    // Add form submission handling
    const simulationForm = document.getElementById('simulationForm');
    if (simulationForm) {
        simulationForm.addEventListener('submit', function (event) {
            // Show loading indicator or message if desired
            console.log('Form submitted, waiting for response...');
        });
    }

    // Handle option selection for enabling Start button
    const startButton = document.getElementById('startButton');
    const radioOptions = document.querySelectorAll('input[name="option"]');
    const toggleMetrics = (showTraining, showHyperparameters, showReward) => {
        document.getElementById("training-metrics").style.display = showTraining ? "block" : "none";
        document.getElementById("hyperparameters-metrics").style.display = showHyperparameters ? "block" : "none";
        document.getElementById("reward-metrics").style.display = showReward ? "block" : "none";
    };

    if (startButton && radioOptions.length > 0) {
        // Function to check if any option is selected
        function checkOptionSelection() {
            const isOptionSelected = Array.from(radioOptions).some(radio => radio.checked);
            startButton.disabled = !isOptionSelected;
        }

        // Add event listeners to all radio buttons
        radioOptions.forEach(radio => {
            radio.addEventListener('change', checkOptionSelection);
        });

        // Set up display toggle for general metrics section
        const trainModel = document.getElementById("trainmodel");
        const testModel = document.getElementById("testmodel");

        if (trainModel && testModel) {
            trainModel.addEventListener("click", () => {
                toggleMetrics(true, true, true);
            });

            testModel.addEventListener("click", () => {
                toggleMetrics(false, false, true);
            });

        }
    }

    // Set up reward parameter sliders
    const sliderIds = ['waiting_time_penalty', 'throughput_reward', 'journey_reward', 'queue_penalty'];
    sliderIds.forEach(id => {
        const slider = document.getElementById(id);
        if (slider) {
            slider.addEventListener('input', function () {
                document.getElementById(`${id}_value`).textContent = this.value;
            });
        }
    });
});
